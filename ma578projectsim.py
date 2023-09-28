from collections import namedtuple
import jax
import jax.numpy as jnp
from jax import grad, random
from jax.scipy.stats import gaussian_kde
from tqdm import trange
import time
# norm along feature dimension d in a data with shape (N,d)
_norm = lambda x: jnp.sqrt(jnp.sum(jnp.square(x),axis=1))

# likelihood distribution
def loglikeli(r,scale,data):

    c = (0.5/(scale*jnp.sqrt(2*jnp.pi))) * jnp.exp(-((_norm(data) - r[0]) ** 2) / (2 * scale ** 2)) + \
        (0.5/(scale*jnp.sqrt(2*jnp.pi))) * jnp.exp(-((_norm(data) - r[1]) ** 2) / (2 * scale ** 2))

    log_likelihood = jnp.sum(jnp.log(c))
    return log_likelihood
grad_loglikeli = grad(loglikeli)

# prior distribution
def logprior(r,scale=10):
    return jnp.log(1/(2*jnp.pi*scale**2)) * ( -r[0]**2/(2*scale**2) - r[1]**2/(2*scale**2))
grad_logprior = grad(logprior)

# ring mixture sampler
def sampler(r,scale,N):
    pi = 3.141592653

    r1 = random.truncated_normal(random.PRNGKey(1), -r[0],jnp.inf, shape=(N,)) * scale + r[0]
    r2 = random.truncated_normal(random.PRNGKey(2), -r[1],jnp.inf, shape=(N,)) * scale + r[1]
    r_mix = jnp.stack((r1,r2),axis=1)
    mixture = random.categorical(random.PRNGKey(3),jnp.array([1.,1.]),shape=(N,))
    print(r_mix.shape)
    r_s = r_mix[jnp.arange(len(r_mix)),mixture]
    theta = jax.random.uniform(random.PRNGKey(4), shape=(N,)) * 2 * pi


    xcoords = r_s * jnp.cos(theta)
    ycoords = r_s * jnp.sin(theta)
    return jnp.stack([xcoords, ycoords], axis=1)


# sample points from the distribution
# 10000 points
data = sampler(jnp.array([1,2]),scale=0.2,N=10000)

#Plot sampled data
import matplotlib.pyplot as plt
x,y = data[:,0],data[:,1]
xy = jnp.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=50,marker='.')
ax.set_aspect('equal')
plt.show()

#SGLD
data_size = len(data)
batch_size = 100
batch_per_epoch = data_size // batch_size
epoch_num = 100

para = jnp.array([0.1,0.1])
para_list_1 = []
para_list_1.append(para)
step_count = 0
st = time.time()
for i in trange(epoch_num):
    random.shuffle(random.PRNGKey(i), data, axis=0)
    for j in range(batch_per_epoch):
        batch_data = data[j*batch_size:(j+1)*batch_size]
        step_size = 0.00001*(0.0001+step_count+1)**(-0.55)
        mygrad = (data_size / batch_size) * grad_loglikeli(para,scale=0.2,data=batch_data) + grad_logprior(para)
        para = para + 0.5*step_size * mygrad + jnp.sqrt(step_size) * random.normal(random.PRNGKey(step_count),shape=(2,))
        para_list_1.append(para)
        print(para)
        step_count += 1
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

#pSGLD
data_size = len(data)
batch_size = 100
batch_per_epoch = data_size // batch_size
epoch_num = 100

para = jnp.array([0.1,0.1])
para_list_2 = []
para_list_2.append(para)
step_count = 0
st = time.time()
for i in range(epoch_num):
    random.shuffle(random.PRNGKey(i), data, axis=0)
    for j in range(batch_per_epoch):
        batch_data = data[j*batch_size:(j+1)*batch_size]

        step_size = 0.00001*(0.0001+step_count+1)**(-0.55)
        mygrad = (data_size / batch_size) * grad_loglikeli(para,scale=0.2,data=batch_data) + grad_logprior(para)

        if step_count==0:
            exps_sqgrad = jnp.square(mygrad)
        else:
            exps_sqgrad = 0.9999*exps_sqgrad + 0.0001*jnp.square(mygrad)

        M = (jnp.sqrt(exps_sqgrad) + 0.0001)
        para = para + 0.5*step_size * mygrad / M + jnp.sqrt(step_size) * random.normal(random.PRNGKey(step_count),shape=(2,)) / jnp.sqrt(M)
        para_list_2.append(para)
        print(para)

        step_count += 1
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# MALA
data_size = len(data)

para = jnp.array([0.1,0.1])
mygrad = grad_loglikeli(para,scale=0.2,data=data) + grad_logprior(para)
para_list_3 = []
para_list_3.append(para)
step_num = 10000
step_count = 0
st = time.time()
for i in range(step_num):

    step_size = 0.00001*(0.0001+step_count+1)**(-0.55)

    para_pro = para + 0.5*step_size * mygrad + jnp.sqrt(step_size) * random.normal(random.PRNGKey(step_count),shape=(2,))
    mygrad_pro = grad_loglikeli(para_pro,scale=0.2,data=data) + grad_logprior(para_pro)

    A1 = loglikeli(para_pro,scale=0.2,data=data) + logprior(para_pro) - \
         (loglikeli(para,scale=0.2,data=data) + logprior(para))
    A2 = -0.5 * jnp.dot((para_pro - para),mygrad_pro + mygrad)
    A3 = 0.125 * step_size**2 * (jnp.sum(jnp.square(para)) - jnp.sum(jnp.square(para_pro)))
    thres = jnp.exp(A1 + A2 + A3)
    if random.uniform(random.PRNGKey(step_count)) < jnp.minimum(1,thres):
        para = para_pro
        mygrad = mygrad_pro

    para_list_3.append(para)
    print(para)

    step_count += 1
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

import numpy as np
para_list_1 = np.array(para_list_1)
para_list_2 = np.array(para_list_2)
para_list_3 = np.array(para_list_3)

plt.plot(para_list_1[:20,0],label='SGLD')
plt.plot(para_list_2[:20,0],label='pSGLD')
plt.plot(para_list_3[:20,0],label='MALA')
plt.xlabel('Itertation')
plt.ylabel('r1')
plt.legend()

plt.plot(para_list_1[:20,1],label='SGLD')
plt.plot(para_list_2[:20,1],label='pSGLD')
plt.plot(para_list_3[:20,1],label='MALA')
plt.xlabel('Itertation')
plt.ylabel('r2')
plt.legend()

plt.plot(para_list_1[-2000:,0],label='SGLD')
# plt.plot(para_list_2[-100:,0],label='pSGLD')
plt.plot(para_list_3[-2000:,0],label='MALA')
plt.xlabel('Itertation')
plt.ylabel('r1')
plt.legend()

plt.plot(para_list_1[-2000:,1],label='SGLD')
# plt.plot(para_list_2[-1000:,1],label='pSGLD')
plt.plot(para_list_3[-2000:,1],label='MALA')
plt.xlabel('Itertation')
plt.ylabel('r2')
plt.legend()