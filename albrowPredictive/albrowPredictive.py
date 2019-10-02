import numpy as np
from scipy.stats import uniform


class AlbrowPredictive:

    def _init_(times,
               mags,
               sd_mags,
               alert_time):
 
    self.times = times
    self.mags = mags
    self.sd_mags = sd_mags
    self.alert_time = alert_time

    #used to set the uniform prior on base mag
    self.mag_min = np.min(mags)
    self.mag_max = np.max(mags)
    
    #initialized the MAP values of the parameters
    self.MAP_ln_tE = 0.0
    self.MAP_ln_A0 = 0.0
    self.MAP_deltaT = 0.0
    self.MAP_flb = 0.0
    self.MAP_mb = 0.0
 
    
 
    def compute_log_prior(params):
        """Calcuales the the natural log of
        the prior p(params)  for a given set of parameters.
        """
        ln_pr_ln_tE = params[0]
        ln_pr_ln_A0 = params[1]
        ln_pr_deltaT = params[2]
        ln_pr_fbl = params[3]
        ln_pr_mb = params[4]

        # Equation (16)
        ln_pr_ln_tE = np.log(0.476) - (ln_tE-1.333)**2 / 0.330 
        # Equation (15)
        ln_pr_ln_A0 = np.log(0.660) - (-1.289*ln_A0)
        # Equation (17)
        ln_pr_deltaT = np.log(0.156) - (deltaT-1.432)**2 / 0.458
        # Paper doesnt mention the prior used, but I assume it to be uniform
        ln_pr_fbl = uniform.logpdf(fbl,0.0,1.0)
        # Paper doesnr mention the prior used but I will asuumed it to be uniform
        ln_pr_mb = uniform.logpdf(mb,self.mag_min - 1.0, self.mag_max + 1.0)

   
        return ln_pr_fbl + ln_pr_ln_A0 + ln_pr_ln_deltaT + ln_pr_ln_tE + ln_pr_mb


    def compute_log_likelihood(self,params):
        """Calculates the natural log of 
        the likelihood p(data|params) for a given set of paramters
        """
        tE = np.exp(params[0])
        A0 = np.exp(params[1])
        deltaT = params[2]
        fbl = params[3]
        mb = params[4]
        
        
        gen_mag = 

        return 0.0

    def neg_log_likelihood(self,params):
        return -self.compute_log_likelihood(params)                     
   
    def compute_log_prob(self,params):
        """Calculates the the natural log of the p(data,params)
        for the given parameters
        """
        return self.compute_log_prior() + self.compute_log_likelihood() 

    def neg_log_prob(self,paramas):
        return -self.compute_log_prob()


  


         
