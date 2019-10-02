import numpy as np



class AlbrowPredictive:

    def _init_(times,
               flux,
               sd_flux)

    def compute_log_prior(...):
        """Calcuales the the natural log of
        the prior for a given set of parameters.
        """

        # Equation (16)
        ln_pr_ln_tE = np.log(0.476) - (ln_tE-1.333)**2 / 0.330 
        # Equation (15)
        ln_pr_ln_A0 = np.log(0.660) - (-1.289*ln-A0)
        # Equation (17)
        ln_pr_ln_deltaT = np.log(0.156) - (ln_deltaT-1.432)**2 / 0.458
       return 
