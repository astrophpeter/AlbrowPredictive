import numpy as np
from scipy.stats import uniform
from scipy.optimize import minimize

class AlbrowPredictive:

    def __init__(self,times,mags,sd_mags,alert_time):
 
        self.times = times
        self.mags = mags
        self.sd_mags = sd_mags
        self.alert_time = alert_time

        #used to set the uniform prior on base mag
        self.mag_min = np.min(mags)
        self.mag_max = np.max(mags)
    
        #initialized the MAP values of the parameters
        self.MAP_params = np.zeros(5)
    
        #initialize the MLE values of the parameter
        self.MLE_params = np.zeros(5)

        #initial quess for the optimization
        self.initial_params = np.array([20.0,10.0,5.0,0.8,np.mean(mag)])
    
 
    def compute_log_prior(self,params):
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
        pred_mag  = self._pred_mag(params,self.times)
        sigma_2 = sd_mags**2         
        log_likelihood = -0.5*np.sum((pred_mag - mags)**2 / sigma_2+ np.log(sigma_2))

        return log_likelihood

    def neg_log_likelihood(self,params):
        return -self.compute_log_likelihood(params)                     
   
    def compute_log_prob(self,params):
        """Calculates the the natural log of the p(data,params)
        for the given parameters
        """
        return self.compute_log_prior() + self.compute_log_likelihood() 

    def neg_log_prob(self,paramas):
        return -self.compute_log_prob()

    def train_MAP(self):
        """Finds the Maximum A-prior estimate of the
        model parameters
        """
        
        result = minimize(self.neg_log_prob,x0=self.initial_params)
        self.MAP_params = result['x']
        print(result)


    def train_MLE(self):
        """Finds the maximum likelihood estimate of the
        model parameters
        """

        result = minimize(self.neg_log_likelihood,x0=self.initial_params)
        self.MLE_params = results['x']

    def predict_MAP(self,new_times):
        """Predict magnitudes at the times new times using the
        MAP paramters
        """
        return self._pred_mag(self.MAP_params,new_times)

    def predict_MLE(self,new_times):
        """Predict magnitued at the times new times using
        the MLE paramters
        """
        return self._pred_mag(self.MLE_params,new_times)


    def _pred_mag(self,params,times):
        """Computes the magnitude for the PSPL model at given times,
        and with model parameters.
        """
        tE = np.exp(params[0])
        A0 = np.exp(params[1])
        deltaT = params[2]
        fbl = params[3]
        mb = params[4]


        u0 = np.sqrt((2*A0/np.sqrt(A0**2-1))-2)
        u = np.sqrt(u0**2+((self.times-deltaT-self.alert_time)/tE)**2)
        Amp = (u**2+2) / (u*np.sqrt(u**2+4))

        pred_mag  = mb - 2.5*np.log10(Amp+fbl)

        return pred_mag

    def find_alert_time(self):
        """Find the alert time in the data. This is defined in the paper
        as the time where there are 3 data points 1 standard deviation away from
        baseline.
        """
        

         
