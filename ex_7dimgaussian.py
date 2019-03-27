#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:35:03 2019

@author: duttar
"""
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# create a bimodal bivariate normal distribution

mu1 = np.array([2,12,3,12,4,11,3])
sigma1 = np.array([2,1,2,1,1,2,1])
mvn1 = multivariate_normal(mu1,sigma1) 

mu2 = np.array([10,6,12,4,11,2,10])
sigma2 = np.array([1,2,1,2,2,1,2])
mvn2 = multivariate_normal(mu2,sigma2)

postt = lambda x: np.log((mvn1.pdf(x)+mvn2.pdf(x))/ \
                         (mvn1.pdf(mu1)+mvn2.pdf(mu2))) 

#%%
from collections import namedtuple
import matplotlib.pyplot as plt
runfile('SMC.py', wdir='.')

# create tuple object with desired number of markov chains, chain length and 
# lower bound and upper bound of the model
NT1 = namedtuple('NT1', 'N Neff target LB UB')
opt = NT1(2000, 50, postt, np.array([0,0,0,0,0,0,0]), \
          np.array([15,15,15,15,15,15,15]))

# tuple object for the samples
NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')
samples = NT2(None, None, None, None, None, None)

# run the SMC sampling
final = SMC_samples(opt,samples, NT1, NT2)

# plot the results: histograms 
plt.subplot(2, 4, 1)
n, bins, patches = plt.hist(final.allsamples[:,0], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 1')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2, 4, 2)
n, bins, patches = plt.hist(final.allsamples[:,1], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 2')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2, 4, 3)
n, bins, patches = plt.hist(final.allsamples[:,2], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 3')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2, 4, 4)
n, bins, patches = plt.hist(final.allsamples[:,3], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 4')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2, 4, 5)
n, bins, patches = plt.hist(final.allsamples[:,4], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 5')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2, 4, 6)
n, bins, patches = plt.hist(final.allsamples[:,5], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 6')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2, 4, 7)
n, bins, patches = plt.hist(final.allsamples[:,6], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.xlabel('Par 7')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)
plt.show()