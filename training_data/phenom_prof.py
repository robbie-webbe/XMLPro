#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:33:45 2024

@author: rwebbe
"""

import numpy as np
from astropy.modeling import models


def QPEProfile(length,amplitude,width,dutycycle):
    '''
    Function for the production of QPE profiles for simulated lightcurves.
    
    Parameters
    length - Length of QPE profile, in 250s time bins
    amplitude - The relative magnitude of the eruptions comparative to their baselines.
    width - The FWHM of the eruptions
    dutycycle - The fraction of the time between successive eruptions as compared with the eruption duration.
    
    Returns
    profile - Repeated eruption profile for the given parameters. A Gaussian is assumed for the profile.
                
    '''
    
    #set up a time array for QPE modeling and a baseline of 1 for the quiescent periods.
    x = np.arange(0,length*250,250)
    model = models.Const1D(1)
    
    #determine the recurrence time
    t_rec = width / dutycycle
    
    #determine the number of eruptions to be included within the lightcurve
    if (length-1) * 250 % t_rec == 0:
        no_eruptions = (((length-1) * 250) // t_rec) + 2
    else:
        no_eruptions = (((length-1) * 250) // t_rec) + 3
    
    #create a series of eruptions, with the first at t=0
    eruption_times = np.arange(0,t_rec * no_eruptions +1,t_rec,dtype=float)
    #shift the eruptions by a random fraction of the recurrence time
    eruption_times -= t_rec * np.random.uniform(0,1)
    
    #add the eruption profiles to the baseline
    for i in eruption_times:
        model += models.Gaussian1D(amplitude=amplitude,mean=i,stddev=width/(2*np.sqrt(2*np.log(2))))
    
    #create the overall profile
    profile = model(x)
    
    return profile
    