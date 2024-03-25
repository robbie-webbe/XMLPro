#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:58:02 2024

@author: rwebbe
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,10)
sys.path.append(os.getcwd()+'/training_data/')

from scipy.stats import poisson, exponnorm
from LCG import ELC_model, QPEProfile

'''
Script to enable the running of Emmanoulopoulos 2013 script for the generation of lightcurves. For each class of
lightcurve to be generated there are default parameters for some of the values given below for each class but
these can be amended and new classes added if required.

Valid types for the phenomena to be generated currently include:
    QPE - Quasi-Periodic Eruption
    QPO - Quasi-Periodic Oscillation
    IF - Isolated Flare
    CV - Cataclysmic Variable
    PON - Poissonian Noise
    
Types where this functionality is being added include:
    EB - Eclipsing Binary
    GRB - Gamma Ray Burst
    T1B - Type 1 X-ray Burst
    
Over time these classes will be added as their profiles are understood. These strings must be used exactly as given
as the first argument for the terminal command. The second argument will be 0 (without phenomena) or 1 (with phenomena).

'''

#determine the input phenomenon type and presence
phenom_name = sys.argv[1]
phenom_present = sys.argv[2]

#determine the type of phenomenon from the input
if phenom_name not in ['QPE','QPO','IF','CV','PON']:
    print("That phenomenon is not yet supported. Try from the approved list (QPE, QPO, IF, CV, PON) if appropriate.")
    sys.exit()

#define the number of lightcurves to be simulated
no_lcs = 10000
#no_lcs=10
   
#generate lightcurves matching the length, time binning and features required for the phenomenon of interest
if phenom_name == 'QPE':
    if phenom_present == '1':
        
        #if eruptions are present, start with an AGN baseline
        #for QPEs time binning of 250s is acceptable, and durations of 120ks (although they may be truncated during feature extraction)
        times = np.arange(0,481*250,250,dtype=float)
        
        #draw a variety of different values for the underlying TK95 power law. These values are following the 
        #distribution identified by Gonzalez-Martin
        pl_soft = 2.06 + 0.01*np.random.randn(no_lcs)
        pl_hard = 1.77 + 0.01*np.random.randn(no_lcs)
        
        #simulate a variety of values of the lognormal distribution underlying the AGN lightcurves
        s_vals = np.random.uniform(0.05,0.5,size=no_lcs)
        mean_vals = np.random.uniform(0.0,0.5,size=no_lcs)
        scale_vals = np.random.uniform(0.0001,0.5,size=no_lcs)
        
        #sample the QPE characteristics for their features
        amps = abs(exponnorm.rvs(3.2210,2.878,2.56652,size=no_lcs))
        durs = abs(exponnorm.rvs(1234.7,696.9,0.77211,size=no_lcs))
        DCs = abs(exponnorm.rvs(1.9698,0.05547,0.01442,size=no_lcs))
        
        #simulate a series of Emmanoulopoulos lightcurves
        soft_fluxes = np.zeros((no_lcs,481))
        hard_fluxes = np.zeros((no_lcs,481))
        for i in range(no_lcs):
            #simulate the base Emman. lc
            soft_base = ELC_model(120000,250,pl_soft[i],'lognorm',PDF_args={'s':s_vals[i],'loc':mean_vals[i],'scale':0.5*scale_vals[i]})
            hard_time, hard_base = ELC_model(120000,250,pl_hard[i],'lognorm',PDF_args={'s':np.random.uniform()*s_vals[i],'loc':np.random.uniform()*mean_vals[i],'scale':np.random.uniform()*scale_vals[i]})
            eruptions_profile = QPEProfile(length=481,amplitude=amps[i],width=durs[i],dutycycle=DCs[i])
            #convolve the eruptions with the baseline to create the QPE lightcurve
            qpe_soft = soft_base[1] * eruptions_profile
            #convert from rate to counts, with a possible factor to make the lightcurves appear fainter,
            #due to distance, extinction or other observational effects.
            dim_factor = np.random.uniform(0.1,1)
            qpe_soft *= (250*dim_factor)
            hard_base *= (250*dim_factor)
            #add poissonian noise to the result
            for k in range(len(qpe_soft)):
                qpe_soft[k] = poisson.rvs(mu=qpe_soft[k])
                hard_base[k] = poisson.rvs(mu=hard_base[k])
            #and revert to count rates
            qpe_soft /= 250
            hard_base /= 250
            
            #add to the output array
            soft_fluxes[i,:] = qpe_soft
            hard_fluxes[i,:] = hard_base
        
        softfile_location = 'QPE/QPE_sim_lcsoft_dt250.csv'
        hardfile_location = 'QPE/QPE_sim_lchard_dt250.csv'
        fullfile_location = 'QPE/QPE_sim_lcfull_dt250.csv'
        
    elif phenom_present == '0':
        #if eruptions are present, start with an AGN baseline
        #for QPEs time binning of 250s is acceptable, and durations of 120ks (although they may be truncated during feature extraction)
        times = np.arange(0,481*250,250,dtype=float)
        
        #draw a variety of different values for the underlying TK95 power law. These values are following the 
        #distribution identified by Gonzalez-Martin
        pl_soft = 2.06 + 0.01*np.random.randn(no_lcs)
        pl_hard = 1.77 + 0.01*np.random.randn(no_lcs)
        
        #simulate a variety of values of the lognormal distribution underlying the AGN lightcurves
        s_vals = np.random.uniform(0.01,0.5,size=no_lcs)
        mean_vals = np.random.uniform(0.0,0.5,size=no_lcs)
        scale_vals = np.random.uniform(0.0001,0.5,size=no_lcs)
        
        #simulate a series of Emmanoulopoulos lightcurves
        soft_fluxes = np.zeros((no_lcs,481))
        hard_fluxes = np.zeros((no_lcs,481))
        for i in range(no_lcs):
            #simulate the base Emman. lc
            soft_time, soft_base = ELC_model(120000,250,pl_soft[i],'lognorm',PDF_args={'s':s_vals[i],'loc':mean_vals[i],'scale':scale_vals[i]})
            hard_time, hard_base = ELC_model(120000,250,pl_hard[i],'lognorm',PDF_args={'s':s_vals[i],'loc':10**(np.random.uniform(-1,1))*mean_vals[i],'scale':10**(np.random.uniform(-1,1))*scale_vals[i]})
            #convert from rate to counts, with dimming factor
            dim_factor = np.random.uniform(0.1,1)
            soft_base *= (250*dim_factor)
            hard_base *= (250*dim_factor)
            #add poissonian noise to the result
            for k in range(len(soft_base)):
                soft_base[k] = poisson.rvs(mu=soft_base[k])
                hard_base[k] = poisson.rvs(mu=hard_base[k])
            #and revert to count rates
            soft_base /= 250
            hard_base /= 250
            
            #add to the output array
            soft_fluxes[i,:] = soft_base
            hard_fluxes[i,:] = hard_base
        
        softfile_location = 'QPE/nPE_sim_lcsoft_dt250.csv'
        hardfile_location = 'QPE/nPE_sim_lchard_dt250.csv'
        fullfile_location = 'QPE/nPE_sim_lcfull_dt250.csv'
        
        
if phenom_name == 'QPO':
    if phenom_present == '1':
        
        #if an oscillation is present, start with an AGN/XRB baseline
        #for QPOs time binning of 100s is acceptable, and durations of 120ks (although they may be truncated during feature extraction)
        times = np.arange(0,1201*100,100,dtype=float)
        
        #draw a variety of different values for the underlying TK95 power law. These values are following the 
        #distribution identified by Gonzalez-Martin
        pl_soft = 2.06 + 0.01*np.random.randn(no_lcs)
        pl_hard = 1.77 + 0.01*np.random.randn(no_lcs)
        
        #simulate a variety of values of the lognormal distribution underlying the AGN lightcurves
        s_vals = np.random.uniform(0.05,0.5,size=no_lcs)
        mean_vals = np.random.uniform(0.0,0.5,size=no_lcs)
        scale_vals = np.random.uniform(0.0001,0.5,size=no_lcs)
        
        #determine the QPO fractional variability, and timescales
        QPO_period = np.random.uniform(3600,86400,size=no_lcs)
        QPO_var = np.random.uniform(0.05,0.50,size=no_lcs)
        QPO_phase = np.random.uniform(0,2*np.pi,size=no_lcs)
        
        #simulate a series of Emmanoulopoulos lightcurves
        soft_fluxes = np.zeros((no_lcs,1201))
        hard_fluxes = np.zeros((no_lcs,1201))
        for i in range(no_lcs):
            #simulate the base Emman. lc
            soft_time, soft_base = ELC_model(120000,100,pl_soft[i],'lognorm',PDF_args={'s':s_vals[i],'loc':mean_vals[i],'scale':scale_vals[i]})
            hard_time, hard_base = ELC_model(120000,100,pl_hard[i],'lognorm',PDF_args={'s':s_vals[i],'loc':mean_vals[i],'scale':scale_vals[i]})
            
            #find the rms variability as a fraction of the average flux
            soft_rms_frac = np.std(soft_base)/np.average(soft_base)
            hard_rms_frac = np.std(hard_base)/np.average(hard_base)
            
            #determine the QPO RMS as a fraction of the total RMS
            QPO_amp_soft = soft_rms_frac * QPO_var[i]
            QPO_amp_hard = hard_rms_frac * QPO_var[i]
            
            #generate the QPO signals in the two bands
            QPO_signal_soft = 1 + QPO_amp_soft * np.sin(2*np.pi*((times/QPO_period[i])-QPO_phase[i]))
            QPO_signal_hard = 1 + QPO_amp_hard * np.sin(2*np.pi*((times/QPO_period[i])-(QPO_phase[i]+np.random.uniform(-0.1,0.1))))
            
            #convolve the oscillation with the baselines to create the QPO lightcurves
            qpo_soft = soft_base * QPO_signal_soft
            qpo_hard = hard_base * QPO_signal_hard
            #convert from rate to counts, with a possible factor to make the lightcurves appear fainter,
            #due to distance, extinction or other observational effects.
            dim_factor = np.random.uniform(0.1,1)
            qpo_soft *= (100*dim_factor)
            qpo_hard *= (100*dim_factor)
            #add poissonian noise to the result
            for k in range(len(qpo_soft)):
                qpo_soft[k] = poisson.rvs(mu=qpo_soft[k])
                qpo_hard[k] = poisson.rvs(mu=qpo_hard[k])
            #and revert to count rates
            qpo_soft /= 100
            qpo_hard /= 100
            
            #add to the output array
            soft_fluxes[i,:] = qpo_soft
            hard_fluxes[i,:] = qpo_hard
        
        softfile_location = 'QPO/QPO_sim_lcsoft_dt100.csv'
        hardfile_location = 'QPO/QPO_sim_lchard_dt100.csv'
        fullfile_location = 'QPO/QPO_sim_lcfull_dt100.csv'
        
    elif phenom_present == '0':
        #if eruptions are present, start with an AGN baseline
        #for QPEs time binning of 250s is acceptable, and durations of 120ks (although they may be truncated during feature extraction)
        times = np.arange(0,1201*100,100,dtype=float)
        
        #draw a variety of different values for the underlying TK95 power law. These values are following the 
        #distribution identified by Gonzalez-Martin
        pl_soft = 2.06 + 0.01*np.random.randn(no_lcs)
        pl_hard = 1.77 + 0.01*np.random.randn(no_lcs)
        
        #simulate a variety of values of the lognormal distribution underlying the AGN lightcurves
        s_vals = np.random.uniform(0.01,0.5,size=no_lcs)
        mean_vals = np.random.uniform(0.0,0.5,size=no_lcs)
        scale_vals = np.random.uniform(0.0001,0.5,size=no_lcs)
        
        #simulate a series of Emmanoulopoulos lightcurves
        soft_fluxes = np.zeros((no_lcs,1201))
        hard_fluxes = np.zeros((no_lcs,1201))
        for i in range(no_lcs):
            #simulate the base Emman. lc
            soft_time, soft_base = ELC_model(120000,100,pl_soft[i],'lognorm',PDF_args={'s':s_vals[i],'loc':mean_vals[i],'scale':scale_vals[i]})
            hard_time, hard_base = ELC_model(120000,100,pl_hard[i],'lognorm',PDF_args={'s':s_vals[i],'loc':10**(np.random.uniform(-1,1))*mean_vals[i],'scale':10**(np.random.uniform(-1,1))*scale_vals[i]})
            #convert from rate to counts, with dimming factor
            dim_factor = np.random.uniform(0.1,1)
            soft_base *= (100*dim_factor)
            hard_base *= (100*dim_factor)
            #add poissonian noise to the result
            for k in range(len(soft_base)):
                soft_base[k] = poisson.rvs(mu=soft_base[k])
                hard_base[k] = poisson.rvs(mu=hard_base[k])
            #and revert to count rates
            soft_base /= 100
            hard_base /= 100
            
            #add to the output array
            soft_fluxes[i,:] = soft_base
            hard_fluxes[i,:] = hard_base
        
        softfile_location = 'QPO/nPO_sim_lcsoft_dt100.csv'
        hardfile_location = 'QPO/nPO_sim_lchard_dt100.csv'
        fullfile_location = 'QPO/nPO_sim_lcfull_dt100.csv'

if phenom_name == 'IF':
    if phenom_present == '1':

        # if an isolated flare is present, start with a low-level mildly variable baseline just above poisson noise levels, or higher.
        # for IFs, in order to accomodate transient events, time binning of 10s is acceptable, and durations of 120ks (although they
        # may be truncated during feature extraction)
        times = np.arange(0, 12001 * 10, 10, dtype=float)

        # draw a variety of different values for the underlying TK95 power law. Given the wide variety of possible hosts
        # these can take a broad variety of values.
        pl_index = np.abs(np.random.normal(0,2,size=no_lcs))

        # simulate a variety of values of the poissonian distribution underlying the random source lightcurves
        s_vals = np.random.uniform(0.05, 0.5, size=no_lcs)
        mean_vals = np.random.uniform(0.0, 0.5, size=no_lcs)
        scale_vals = np.random.uniform(0.0001, 0.5, size=no_lcs)

        # determine the QPO fractional variability, and timescales
        QPO_period = np.random.uniform(3600, 86400, size=no_lcs)
        QPO_var = np.random.uniform(0.05, 0.50, size=no_lcs)
        QPO_phase = np.random.uniform(0, 2 * np.pi, size=no_lcs)

        # simulate a series of Emmanoulopoulos lightcurves
        soft_fluxes = np.zeros((no_lcs, 1201))
        hard_fluxes = np.zeros((no_lcs, 1201))
        for i in range(no_lcs):
            # simulate the base Emman. lc
            soft_time, soft_base = ELC_model(120000, 100, pl_soft[i], 'lognorm',
                                             PDF_args={'s': s_vals[i], 'loc': mean_vals[i], 'scale': scale_vals[i]})
            hard_time, hard_base = ELC_model(120000, 100, pl_hard[i], 'lognorm',
                                             PDF_args={'s': s_vals[i], 'loc': mean_vals[i], 'scale': scale_vals[i]})

            # find the rms variability as a fraction of the average flux
            soft_rms_frac = np.std(soft_base) / np.average(soft_base)
            hard_rms_frac = np.std(hard_base) / np.average(hard_base)

            # determine the QPO RMS as a fraction of the total RMS
            QPO_amp_soft = soft_rms_frac * QPO_var[i]
            QPO_amp_hard = hard_rms_frac * QPO_var[i]

            # generate the QPO signals in the two bands
            QPO_signal_soft = 1 + QPO_amp_soft * np.sin(2 * np.pi * ((times / QPO_period[i]) - QPO_phase[i]))
            QPO_signal_hard = 1 + QPO_amp_hard * np.sin(
                2 * np.pi * ((times / QPO_period[i]) - (QPO_phase[i] + np.random.uniform(-0.1, 0.1))))

            # convolve the oscillation with the baselines to create the QPO lightcurves
            qpo_soft = soft_base * QPO_signal_soft
            qpo_hard = hard_base * QPO_signal_hard
            # convert from rate to counts, with a possible factor to make the lightcurves appear fainter,
            # due to distance, extinction or other observational effects.
            dim_factor = np.random.uniform(0.1, 1)
            qpo_soft *= (100 * dim_factor)
            qpo_hard *= (100 * dim_factor)
            # add poissonian noise to the result
            for k in range(len(qpo_soft)):
                qpo_soft[k] = poisson.rvs(mu=qpo_soft[k])
                qpo_hard[k] = poisson.rvs(mu=qpo_hard[k])
            # and revert to count rates
            qpo_soft /= 100
            qpo_hard /= 100

            # add to the output array
            soft_fluxes[i, :] = qpo_soft
            hard_fluxes[i, :] = qpo_hard

        softfile_location = 'QPO/QPO_sim_lcsoft_dt100.csv'
        hardfile_location = 'QPO/QPO_sim_lchard_dt100.csv'
        fullfile_location = 'QPO/QPO_sim_lcfull_dt100.csv'

    elif phenom_present == '0':
        # if eruptions are present, start with an AGN baseline
        # for QPEs time binning of 250s is acceptable, and durations of 120ks (although they may be truncated during feature extraction)
        times = np.arange(0, 1201 * 100, 100, dtype=float)

        # draw a variety of different values for the underlying TK95 power law. These values are following the
        # distribution identified by Gonzalez-Martin
        pl_soft = 2.06 + 0.01 * np.random.randn(no_lcs)
        pl_hard = 1.77 + 0.01 * np.random.randn(no_lcs)

        # simulate a variety of values of the lognormal distribution underlying the AGN lightcurves
        s_vals = np.random.uniform(0.01, 0.5, size=no_lcs)
        mean_vals = np.random.uniform(0.0, 0.5, size=no_lcs)
        scale_vals = np.random.uniform(0.0001, 0.5, size=no_lcs)

        # simulate a series of Emmanoulopoulos lightcurves
        soft_fluxes = np.zeros((no_lcs, 1201))
        hard_fluxes = np.zeros((no_lcs, 1201))
        for i in range(no_lcs):
            # simulate the base Emman. lc
            soft_time, soft_base = ELC_model(120000, 100, pl_soft[i], 'lognorm',
                                             PDF_args={'s': s_vals[i], 'loc': mean_vals[i], 'scale': scale_vals[i]})
            hard_time, hard_base = ELC_model(120000, 100, pl_hard[i], 'lognorm', PDF_args={'s': s_vals[i],
                                                                                           'loc': 10 ** (
                                                                                               np.random.uniform(-1,
                                                                                                                 1)) *
                                                                                                  mean_vals[i],
                                                                                           'scale': 10 ** (
                                                                                               np.random.uniform(-1,
                                                                                                                 1)) *
                                                                                                    scale_vals[i]})
            # convert from rate to counts, with dimming factor
            dim_factor = np.random.uniform(0.1, 1)
            soft_base *= (100 * dim_factor)
            hard_base *= (100 * dim_factor)
            # add poissonian noise to the result
            for k in range(len(soft_base)):
                soft_base[k] = poisson.rvs(mu=soft_base[k])
                hard_base[k] = poisson.rvs(mu=hard_base[k])
            # and revert to count rates
            soft_base /= 100
            hard_base /= 100

            # add to the output array
            soft_fluxes[i, :] = soft_base
            hard_fluxes[i, :] = hard_base

        softfile_location = 'QPO/nPO_sim_lcsoft_dt100.csv'
        hardfile_location = 'QPO/nPO_sim_lchard_dt100.csv'
        fullfile_location = 'QPO/nPO_sim_lcfull_dt100.csv'
        

#create the output file location and output array
soft_array = np.zeros((no_lcs+1,len(times)))
soft_array[0] = times
soft_array[1:,:] = soft_fluxes
hard_array = np.zeros((no_lcs+1,len(times)))
hard_array[0] = times
hard_array[1:,:] = hard_fluxes
full_array = np.zeros((no_lcs+1,len(times)))
full_array[0] = times
full_array[1:,:] = soft_array[1:,:] + hard_array[1:,:]

#output the file to the desired location
np.savetxt('training_data/'+softfile_location,soft_array,delimiter=',')
np.savetxt('training_data/'+hardfile_location,hard_array,delimiter=',')
np.savetxt('training_data/'+fullfile_location,full_array,delimiter=',')

#plot the first five from the sample in order to check reasonableness
for i in range(10):
    plt.plot(soft_array[0],soft_array[i+1],color='r')
    plt.plot(hard_array[0],hard_array[i+1],color='b')
    plt.plot(full_array[0],full_array[i+1],color='k')
    plt.show()

