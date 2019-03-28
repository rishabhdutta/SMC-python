# SMC-python
Python codes for Sequential Monte Carlo sampling Technique. 

This technique is robust in sampling close to 1000-dimensions posterior probability 
densities.

Uses `numpy`, `scipy` and `collections` libraries. 

## Demonstration 
### Example 1: Bivariate bimodal Gaussian probability density

We draw samples from a bivariate bimodal Gaussian probability density as shown below:   
![Image of Yaktocat](https://github.com/rishabhdutta/SMC-python/blob/master/figures/Figure_1.png)
The two modes for first parameter (`par 1`) is at `[2 10]`, while for the second
parameter (`par 2`) is at `[6 12]`. The corresponding standard deviation are 
`[2 1]` for `par 1` and `[2 1]` for `par 2`, respectively. 

##### codes to calculate target density for the two parameters
`postt` calculates the log of the target distribution.
```
import numpy as np
from scipy.stats import multivariate_normal

mu1 = np.array([2,12])
sigma1 = np.array([2,1])
mvn1 = multivariate_normal(mu1,sigma1) 

mu2 = np.array([10,6])
sigma2 = np.array([1,2])
mvn2 = multivariate_normal(mu2,sigma2)

postt = lambda x: np.log((mvn1.pdf(x)+mvn2.pdf(x))/ \
                         (mvn1.pdf(mu1)+mvn2.pdf(mu2))) 
```
 
##### Generate samples corresponding to the target distribution
Import the `collections` library.  
```
from collections import namedtuple
```
Create two named tuple objects. `opt` corresponds to input parameters of the 
sampling technique. `opt.N` is the number of Markov chains, `opt.Neff` is the 
chain length, `opt.LB` is the lower bound of the target distribution parameters,
and `opt.UB` is the upper bound.  

`samples` corresponds to the information of the samples at each intermediate 
stage. `samples.allsamples` is the ensemble of samples at final stage, `samples.postval`
is the log of target distribution values, `samples.stage` is the array of all 
the stages, `samples.beta` is the array of beta parameters, `samples.covsmpl` is
the sample covariance at final stage, `samples.resmpl` is the resampled samples
at the final stage. 
```
NT1 = namedtuple('NT1', 'N Neff target LB UB')
opt = NT1(2000, 50, postt, np.array([0,0]), np.array([15,15]))

NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')
samples = NT2(None, None, None, None, None, None)
```
Run this line to get all the samples. 
```
final = SMC_samples(opt,samples, NT1, NT2)
```
##### Plot the histograms of the resulting samples
```
plt.subplot(1, 2, 1)
n, bins, patches = plt.hist(final.allsamples[:,0], 50, density=True, \
                            facecolor='b', alpha=0.75)
plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(final.allsamples[:,1], 50, density=True, \
                            facecolor='b', alpha=0.75)
```                        
 
 