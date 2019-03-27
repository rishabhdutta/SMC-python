#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:20:09 2019

@author: duttar
"""

import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# create a bimodal bivariate normal distribution

mu1 = np.array([2,12])
sigma1 = np.array([2,1])
mvn1 = multivariate_normal(mu1,sigma1) 

mu2 = np.array([10,6])
sigma2 = np.array([1,2])
mvn2 = multivariate_normal(mu2,sigma2)

postt = lambda x: np.log((mvn1.pdf(x)+mvn2.pdf(x))/ \
                         (mvn1.pdf(mu1)+mvn2.pdf(mu2))) 

'''
plot the posterior 
'''
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 15.2, 0.2)
Y = np.arange(15.2, 0, -0.2)
XX, YY = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.zeros([XX.shape[0],XX.shape[1]])
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = postt(np.array([XX[i,j],YY[i,j]]))
ZZ = np.exp(Z)

# Plot the surface.
surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-.1, 0.6)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %%
from collections import namedtuple
import matplotlib.pyplot as plt
runfile('SMC.py', wdir='.')

# create tuple object with desired number of markov chains, chain length and 
# lower bound and upper bound of the model
NT1 = namedtuple('NT1', 'N Neff target LB UB')
opt = NT1(2000, 50, postt, np.array([0,0]), np.array([15,15]))

# tuple object for the samples
NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')
samples = NT2(None, None, None, None, None, None)

# run the SMC sampling
final = SMC_samples(opt,samples, NT1, NT2)

# plot the results: histograms 
n, bins, patches = plt.hist(final.allsamples[:,0], 50, density=True, \
                            facecolor='b', alpha=0.75)

plt.xlabel('Par 2')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(final.allsamples[:,1], 50, density=True, \
                            facecolor='b', alpha=0.75)


plt.xlabel('Par 2')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)
plt.show()






























