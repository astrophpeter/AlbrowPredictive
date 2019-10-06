# -*- coding: utf-8 -*-
"""Albrow Predictive.

This module implements the paper: "Early Estimation of Microlensing Event 
Magnifications", by M.D. Albrow 2004:

@ARTICLE{2004ApJ...607..821A,
       author = {{Albrow}, Michael D.},
        title = "{Early Estimation of Microlensing Event Magnifications}",
      journal = {\apj},
     keywords = {Cosmology: Gravitational Lensing, Methods: Data Analysis,
                 Astrophysics},
         year = "2004",
        month = "Jun",
       volume = {607},
       number = {2},
        pages = {821-827},
          doi = {10.1086/383565},
archivePrefix = {arXiv},
       eprint = {astro-ph/0402323},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2004ApJ...607..821A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

All equation references in this module relate to the numbering in the above 
paper.
"""

import numpy as np
from numpy import ndarray
from scipy.stats import uniform
from scipy.optimize import differential_evolution

class AlbrowPredictive:
    """Albrow 2004's predictive microlensing model.
    """

    def __init__(self,times: ndarray, mags: ndarray, sd_mags: ndarray) -> None:
        """Contructs the predictive model using passed in data.

        N = number of data points.
 
        Args:
            times: times, [mjd] units shape = (N,).
            mags: Magnitues, [mag] units, shape = (N,).
            sd_mags: Errors of the magnidudes, [mag] units, shape = (N,)   
        
        """
        #store data
        self.times = times
        self.mags = mags
        self.sd_mags = sd_mags
        
        #find alert time
        self.alert_time = self.find_alert_time()

        #used to set the uniform prior on base mag
        self.mag_min = np.min(mags)
        self.mag_max = np.max(mags)
    
        #initialized the MAP and MLE values of the parameters
        self.map_params = np.zeros(5)
        self.mle_params = np.zeros(5)

        #initial guesses for the optimization
        self.initial_params = np.array([2.0,5.0,7.0,0.8,np.mean(mags)])
        self.initial_params_bounds = [(-1,10),(0.1,8.0),(-1,10),(0.0001,0.999),(self.mag_min,self.mag_max)] 
   
    def compute_log_prior(self,params: ndarray) -> float:
        """Calcuales the the natural log of the prior p(params).  

        This implements the the fixed emperically determined priors from the
        paper.

        Args:
            params : array of model parameters [ln_tE,ln_A0,ln_deltaT,fbl,mb],
                or [ln(Einstien time),ln(Max Amplication),ln(time since alert),
                blending paramerter,baseline magnitude],
                shape = (5,).
        Returns:
            ln_prior : natural logarithm of the prior.
        """
        ln_tE = params[0]
        ln_A0 = params[1]
        ln_deltaT = params[2]
        fbl = params[3]
        mb = params[4]

        # Equation (16,15,17)
        ln_pr_ln_tE = np.log(0.476) - ((ln_tE-1.333)**2 / 0.330) 
        ln_pr_ln_A0 = np.log(0.660) - (1.289*ln_A0)
        ln_pr_ln_deltaT = np.log(0.156) - ((ln_deltaT-1.432)**2 / 0.458)    
     
        # Paper doesnt mention the prior used, but I assume it to be uniform
        ln_pr_fbl = uniform.logpdf(fbl,0.0,1.0)

        # Paper doesnr mention the prior used but I will asuumed it to be uniform
        ln_pr_mb = uniform.logpdf(mb,self.mag_min - 1.0, self.mag_max + 1.0)
 
   
        return ln_pr_fbl + ln_pr_ln_A0 + ln_pr_ln_deltaT + ln_pr_ln_tE + ln_pr_mb


    def compute_log_likelihood(self,params: ndarray) -> float:
        """Calculates the natural log of the likelihood p(data|params). 

        Implements a standard Gaussian Likelihood.

        Args:
            params : array of model parameters [ln_tE,ln_A0,ln_deltaT,fbl,mb],
                or [ln(Einstien time),ln(Max Amplication),ln(time since alert),
                blending paramerter,baseline magnitude],
                shape = (5,).
        Returns:
            ln_likelihood : natural logarithm of the likelihood.
        """
        
        pred_mag  = self._pred_mag(params,self.times)
        sigma_2 = self.sd_mags**2         
        ln_likelihood = -0.5*np.sum((pred_mag - self.mags)**2 / sigma_2+ np.log(sigma_2))

        return ln_likelihood

    def neg_log_likelihood(self,params: ndarray) -> float:
        """Calculates the negative natural log of the likelihood p(data|params). 

        This is for optimizing purposes, for finding the maximum likelihood 
        (mle) solution.

        Args:
            params : array of model parameters [ln_tE,ln_A0,ln_deltaT,fbl,mb],
                or [ln(Einstien time),ln(Max Amplication),ln(time since alert),
                blending paramerter,baseline magnitude],
                shape = (5,).
        Returns:
            neg_ln_likelihood : negative natural logarithm of the likelihood.
        """

        return -self.compute_log_likelihood(params)                     
   
    def compute_log_prob(self,params: ndarray) -> float:
        """Calculates the the natural log of the joint p(data,params).

        This is proportional to the posterior distribution.

        Args:
            params : array of model parameters [ln_tE,ln_A0,ln_deltaT,fbl,mb],
                or [ln(Einstien time),ln(Max Amplication),ln(time since alert),
                blending paramerter,baseline magnitude],
                shape = (5,).
        Returns:
            ln_prob : natural logarithm of the joint model and data distribution.
        """
        return self.compute_log_prior(params) + self.compute_log_likelihood(params) 

    def neg_log_prob(self,params: ndarray) -> float:
        """Calculates the negative of the natural log of p(data,params).
        
        This is for optimizing purposes, for finding the maximum a-posteroir 
        (MAP) solution.
        
        Args:
            params : array of model parameters [ln_tE,ln_A0,ln_deltaT,fbl,mb],
                or [ln(Einstien time),ln(Max Amplication),ln(time since alert),
                blending paramerter,baseline magnitude],
                shape = (5,).
        Returns:
            neg_ln_prob : negative natural logarithm of the joint model and 
                data distribution.
        """
        return -self.compute_log_prob(params)

    def train_MAP(self) -> None:
        """Finds the Maximum a-pr estimate of the
        model parameters
        """
        
        result = differential_evolution(self.neg_log_prob,
                          bounds=self.initial_params_bounds)
        self.map_params = result['x']
        print(result)

    def train_MLE(self) -> None:
        """Finds the maximum likelihood estimate of the
        model parameters
        """

        result = differential_evolution(self.neg_log_likelihood,
                         bounds=self.initial_params_bounds)
        self.mle_params = result['x']

        print(result)

    def predict_MAP(self,new_times):
        """Predict magnitudes at the times new times using the
        MAP paramters
        """
        return self._pred_mag(self.map_params,new_times)

    def predict_MLE(self,new_times):
        """Predict magnitued at the times new times using
        the MLE paramters
        """
        return self._pred_mag(self.mle_params,new_times)
        

    def _pred_mag(self,params,times):
        """Computes the magnitude for the PSPL model at given times,
        and with model parameters.
        """
        tE = 10**params[0]
        A0 = 10**params[1]
        deltaT = 10**params[2]
        fbl = params[3]
        mb = params[4]


        u0 = np.sqrt((2*A0/np.sqrt(A0**2-1))-2)
        u = np.sqrt(u0**2+((times-deltaT-self.alert_time)/tE)**2)
        Amp = (u**2+2) / (u*np.sqrt(u**2+4))

        pred_mag  = mb - 2.5*np.log10(fbl*(Amp-1)+1)

        return pred_mag

    def find_alert_time(self):
        """Find the alert time in the data. This is defined in the paper
        as the time where there are 3 data points 1 standard deviation away from
        baseline.
        """
        
        # Also not clear from the paper how to doe this,
        # use the first 10 data points in the light curve to determine the magnitude
        # baseline

        mean_mag = np.mean(self.mags[:10])
        std_mag  = np.std(self.mags[:10])

        num_above = 0 
        i = 9

        while num_above < 3 and i < len(self.times)-1:
            
            i += 1 

            if self.mags[i] < mean_mag - std_mag:
                num_above += 1
            else:
                num_above = 0.0

        if len(self.times) - 1 == i:
            print("Give me more training data, not alerted yet, this is probably going to fail")
         
        return self.times[i-1] 











     




 
         
