import pytest
import numpy as np

from ..albrowPredictive import AlbrowPredictive

def test_prior():
    """Test if the specific fixed the prior evaluates
    to the correct values
    """

    #dummy data
    mags = np.linspace(1,20,50)
    emags = np.random.random_sample(50)
    times = np.linspace(1,200,50)
    model = AlbrowPredictive(times,mags,emags)
    
    ln_tE = 1.333
    ln_A0 = 3.0
    ln_deltaT = 1.432
    fbl = 0.5
    mb = 10.0

    params = np.array([ln_tE,ln_A0,ln_deltaT,fbl,mb])
    ln_prior = model.compute_log_prior(params)
    
    ans = (np.log(0.660) - 1.289*3 
           + np.log(0.476) 
           + np.log(0.156) 
           + np.log(1.0) 
           + np.log(1.0/(21.0-0.0)))
  
    assert ln_prior == ans
