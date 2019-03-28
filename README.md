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
 