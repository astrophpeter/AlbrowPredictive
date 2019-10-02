import numpy as np
from scipy.stats import uniform


class AlbrowPredictive:

    def _init_(times,
               mag,
               sd_flux,
               alert_time):

    #used to set the uniform prior on base mag
    self.mag_min = np.min(mag)
    self.mag_max = np.max(mag)
    
    #initialized the MAP values of the parameters
    self.MAP_ln_tE = 0.0
    self.MAP_ln_A0 = 0.0
    self.MAP_
 
    def compute_log_prior(...):
        """Calcuales the the natural log of
        the prior for a given set of parameters.
        """

        # Equation (16)
        ln_pr_ln_tE = np.log(0.476) - (ln_tE-1.333)**2 / 0.330 
        # Equation (15)
        ln_pr_ln_A0 = np.log(0.660) - (-1.289*ln_A0)
        # Equation (17)
        ln_pr_ln_deltaT = np.log(0.156) - (ln_deltaT-1.432)**2 / 0.458
        # Paper doesnt mention the prior used, but I assume it to be uniform
        ln_pr_fbl = uniform.logpdf(fbl,0.0,1.0)
        # Paper doesnr mention the prior used but I will asuumed it to be uniform
        ln_pr_mb = uniform.logpdf(mb,self.mag_min - 1.0, self.mag_max + 1.0)

   
        return ln_pr_fbl + ln_pr_ln_A0 + ln_pr_ln_deltaT + ln_pr_ln_tE + ln_pr_mb


    def compute_log_likelihood(..):
        """Calculates the natural log of 
        the likelihood for a given set of paramters
        """

         
