#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:00:48 2019

@author: duttar
"""
import numpy as np

def deterministicR(inIndex,q): 
    """
    Parameters: 
      * inIndex  : index of the posterior values  
      * q  : weight of the posterior values 
      
    Output: 
      * outindx  : index of the resampling  
    """
    
    n_chains = inIndex.shape[0]
    parents = np.arange(n_chains)
    N_childs = np.zeros(n_chains, dtype=int)

    cum_dist = np.cumsum(q)
    aux = np.random.rand(1)
    u = parents + aux
    u /= n_chains
    j = 0
    for i in parents:
        while u[i] > cum_dist[j]:
            j += 1
        N_childs[j] += 1

    indx = 0
    outindx = np.zeros(n_chains, dtype=int)
    for i in parents:
        if N_childs[i] > 0:
            for j in range(indx, (indx + N_childs[i])):
                outindx[j] = parents[i]

        indx += N_childs[i]

    return outindx

def AMH(X,target,covariance,mrun,beta,LB,UB):
    """
    Adaptive Metropolis algorithm
    scales the covariance matrix according to the acceptance rate 
    cov of proposal = (a+ bR)*sigma;  R = acceptance rate
    returns the last sample of the chain
    
    Parameters: 
     *  X : starting model 
     *  target : target distribution (is a function handle and calculates the log posterior)
     *  covariance : covariance of the proposal distribution 
     *  mrun : number of samples 
     *  beta : use the beta parameter of SMC sampling, otherwise 1 
     *  LB : lower bound of the model parameters
     *  UB : upper bound of the model parameters
    
    Outputs: 
     *  G : last sample of the chain 
     *  GP : log posterior value of the last sample 
     *  avg_acc : average acceptance rate
     
     written by : Rishabh Dutta (18 Mar 2019)
     Matlab version written on 12 Mar 2016
     (Don't forget to acknowledge)
    """
    
    Dims = covariance.shape[0]
    logpdf = target(X) 
    V = covariance
    best_P = logpdf * beta 
    P0 = logpdf * beta 
    
    # the following values are estimated empirically 
    a = 1/9
    b = 8/9
    
    sameind = np.where(np.equal(LB, UB))
    
    def dimension(i):
        switcher={
            1:0.441,
            2:0.352,
            3:0.316,
            4:0.285,
            5:0.275,
            6:0.273,
            7:0.270,
            8:0.268,
            9:0.267,
            10:0.266,
            11:0.265,
            12:0.255            
            }
        return switcher.get(i, 0.255)
    
    # set initial scaling factor
    s = a + b*dimension(Dims)
    
    U = np.log(np.random.rand(1,mrun))
    TH = np.zeros((Dims,mrun))
    THP = np.zeros((1,mrun))
    avg_acc = 0 
    factor = np.zeros((1,mrun))
    
    for i in range(mrun):
        X_new = np.random.multivariate_normal(X,s**2*V)
        X_new[sameind] = LB[sameind]
        
        ind1 = np.where(X_new < LB)
        diff1 = LB[ind1] - X_new[ind1]
        X_new[ind1] = LB[ind1] + diff1 
        
        if avg_acc < 0.05: 
            X_new[ind1] = LB[ind1]
            
        ind2 = np.where(X_new > UB)
        diff2 = X_new[ind2] - UB[ind2]
        X_new[ind2] = UB[ind2] - diff2
        
        if avg_acc < 0.05:
            X_new[ind2] = UB[ind2]
            
        P_new = beta * target(X_new)
        
        if P_new > best_P: 
            X = X_new
            best_P = P_new
            P0 = P_new
            acc_rate = 1 
        else:
            rho = P_new - P0 
            acc_rate = np.exp(np.min([0,rho]))
            if U[0,i] <= rho : 
                X = X_new
                P0 = P_new
                
        TH[:,i] = np.transpose(X) 
        THP[0,i] = P0 
        factor[0,i] = s**2
        avg_acc = avg_acc*(i)/(i+1) + acc_rate/(i+1) 
        s = a+ b*avg_acc
        
    G = TH[:,-1]
    GP = THP[0,-1]/beta
    
    return G, GP, avg_acc

# %%
class SMCclass:
    """
    Generates samples of the 'target' posterior PDF using SMC sampling. Also called Adapative Transitional Metropolis
    Importance (sampling) P abbreviated as ATMIP  
    """
    def __init__(self, opt, samples, NT1, NT2, verbose=True):
        """
        Parameters: 
            opt : named tuple 
                - opt.target (lamda function of the posterior)
                - opt.UB (upper bound of parameters)
                - opt.LB (lower bound of parameters)
                - opt.N (number of Markov chains at each stage)
                - opt.Neff (Chain length of the MCMC sampling) 
                
            samples: named tuple
                - samples.allsamples (samples at each stage)
                - samples.postval (log posterior value of samples)
                - samples.beta (array of beta values)
                - samples.stage (array of stages)
                - samples.covsmpl (model covariance at each stage)
                - samples.resmpl (resampled model at each stage)
                
            NT1: create opt object
            NT2: create samples object 
            
        written by: Rishabh Dutta, Dec 12 2018
        (Don't forget to acknowledge)
        
        """
        self.verbose = verbose
        self.opt = opt  
        self.samples = samples
        self.NT1 = NT1
        self.NT2 = NT2
            
    def initialize(self):
        if self.verbose:
            print ("-----------------------------------------------------------------------------------------------")
            print ("-----------------------------------------------------------------------------------------------")
            print(f'Initializing ATMIP with {self.opt.N :8d} Markov chains and {self.opt.Neff :8d} chain length.')
                    
    def prior_samples(self):
        '''
        determines the prior posterior values 
        the prior samples are estimated from lower and upper bounds
        
        Output : samples (NT2 object with estimated posterior values)
        '''
        numpars = self.opt.LB.shape[0]
        diffbnd = self.opt.UB - self.opt.LB
        diffbndN = np.tile(diffbnd,(self.opt.N,1))
        LBN = np.tile(self.opt.LB,(self.opt.N,1))
        
        sampzero = LBN +  np.random.rand(self.opt.N,numpars) * diffbndN
        beta = np.array([0]) 
        stage = np.array([1]) 
        
        postval = np.zeros([self.opt.N,1])
        for i in range(self.opt.N):
            samp0 = sampzero[i,:]
            logpost = self.opt.target(samp0)
            postval[i] = logpost
            
        samples = self.NT2(sampzero, postval, beta, stage, None, None)
        return samples
          
    
    def find_beta(self): 
        """
        Calculates the beta parameter for the next stage
        """
        beta1 = self.samples.beta[-1]       #prev_beta
        beta2 = self.samples.beta[-1]       #prev_beta
        max_post = np.max(self.samples.postval) 
        logpst = self.samples.postval - max_post
        beta = beta1+.5
    
        if beta>1:
            beta = 1
            #logwght = beta.*logpst
            #wght = np.exp(logwght)
    
        refcov = 1 
    
        while beta - beta1 > 1e-6:
            curr_beta = (beta+beta1)/2
            diffbeta = beta-beta1
            logwght = diffbeta*logpst
            wght = np.exp(logwght)
            covwght = np.std(wght)/np.mean(wght)
            if covwght > refcov:
                beta = curr_beta
            else:
                beta1 = curr_beta
            
        betanew = np.min(np.array([1,beta]))
        betaarray = np.append(self.samples.beta,betanew)
        newstage = np.arange(1,self.samples.stage[-1]+2)
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           betaarray, newstage, self.samples.covsmpl, \
                           self.samples.resmpl)
    
        return samples
    
    def resample_stage(self):
        '''
        Resamples the model samples at a certain stage 
        Uses Kitagawa's deterministic resampling algorithm
        '''
        
        # calculate the weight for model samples
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2])* logpst
        wght = np.exp(logwght)
        
        probwght = wght/np.sum(wght)
        inind = np.arange(0,self.opt.N)
        
        outind = deterministicR(inind, probwght)
        newsmpl = self.samples.allsamples[outind,:]
        
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           self.samples.beta, self.samples.stage, \
                           self.samples.covsmpl, newsmpl)
        
        return samples
        
    def make_covariance(self):
        '''
        make the model covariance using the weights and samples from previous 
        stage
        '''
        # calculate the weight for model samples
        
        dims = self.samples.allsamples.shape[1]
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2])* logpst
        wght = np.exp(logwght)
        
        probwght = wght/np.sum(wght)
        weightmat = np.tile(probwght,(1,dims))
        multmat = weightmat * self.samples.allsamples
        
        # calculate the mean samples
        meansmpl = multmat.sum(axis=0, dtype='float')
        
        # calculate the model covariance
        covariance = np.matrix(np.zeros((dims,dims), dtype='float'))
        for i in range(self.opt.N):
            par = self.samples.allsamples[i,:]
            smpldiff = np.matrix(par - meansmpl)
            smpdsq = np.matmul(np.transpose(smpldiff),smpldiff)
            covint = np.multiply(probwght[i], smpdsq)
            covariance += covint
            
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           self.samples.beta, self.samples.stage, \
                           covariance, self.samples.resmpl)
        return samples
        
    def MCMC_samples(self):
        """
        Nothing
        """
        dims = self.samples.allsamples.shape[1]
        
        mhsmpl = np.zeros([self.opt.N,dims])
        mhpost = np.zeros([self.opt.N,1])
        for i in range(self.opt.N):
            start = self.samples.resmpl[i,:]
            G, GP, acc = AMH(start, self.opt.target, self.samples.covsmpl, \
                               self.opt.Neff, self.samples.beta[-1], \
                               self.opt.LB, self.opt.UB)
            mhsmpl[i,:] = np.transpose(G)
            mhpost[i] = GP 
            
        samples = self.NT2(mhsmpl, mhpost, self.samples.beta, \
                           self.samples.stage, self.samples.covsmpl, \
                           self.samples.resmpl)
        return samples
                

# %%
def SMC_samples(opt,samples, NT1, NT2):
    '''
    Sequential Monte Carlo technique
    < a subset of CATMIP by Sarah Minson>
    The method samples the target distribution through several stages (called 
    transitioning of simulated annealing). At each stage the samples corresponds
    to the intermediate PDF between the prior PDF and final target PDF. 
    
    After samples generated at each stage, the beta parameter is generated for
    the next stage. At the next stage, resampling is performed. Then MCMC 
    sampling (adpative Metropolis chains) is resumed from each resampled model. 
    The weigted covariance is estimated using the weights (calculated from 
    posterior values) and samples from previous stage. This procedure is conti-
    nued until beta parameter is 1. 
    
    syntax: output = ATMIP(opt)
    
    Inputs: 
    
        opt : named tuple 
            - opt.target (lamda function of the posterior)
            - opt.UB (upper bound of parameters)
            - opt.LB (lower bound of parameters)
            - opt.N (number of Markov chains at each stage)
            - opt.Neff (Chain length of the MCMC sampling)
            
        samples: named tuple
            - samples.allsamples (samples at an intermediate stage)
            - samples.beta (beta at that stage)
            - samples.postval (posterior values)
            - samples.stage (stage number)
            - samples.covsmpl (model covariance matrix used for MCMC sampling)
    
        NT1 - named tuple structure for opt
        NT2 - named tuple structure for samples
        
    Outputs: 
        
        samples : named tuple
            - samples.allsamples (final samples at the last stage)
            - samples.postval (log posterior values of the final samples)
            - samples.stages (array of all stages)
            - samples.beta (array of beta values)
            - samples.covsmpl (model covariance at final stage)
            - samples.resmpl (resampled model samples at final stage)
            
    written by: Rishabh Dutta, Mar 25 2019
    (Don't forget to acknowledge)
    '''
    current = SMCclass(opt, samples, NT1, NT2)
    current.initialize()            # prints the initialization
    
    if samples.allsamples is None:  # generates prior samples and calculates 
                                    # their posterior values
        print('------Calculating the prior posterior values at stage 1-----')
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.prior_samples()
        
    while samples.beta[-1] != 1:
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.find_beta()            # calculates beta at next stage 
        
        # at next stage here -------------------------------------
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.resample_stage()       # resample the model samples 
        
        # make the model covariance 
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.make_covariance()       
        
        # use the resampled model samples as starting point for MCMC sampling 
        # we use adaptive Metropolis sampling 
        # adaptive proposal is generated using model covariance 
        print(f'Starting metropolis chains at stage = {samples.stage[-1] :3d} and beta = {samples.beta[-1] :.6f}.')
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.MCMC_samples()
    
    return samples
        