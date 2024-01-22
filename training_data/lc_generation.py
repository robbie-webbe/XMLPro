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


from LCG import ELC_model
from phenom_prof import QPEProfile, QPOProfile

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
    
#generate lightcurves matching the length, time binning and features required for the phenomenon of interest
if phenom_name == 'QPE':
    if phenom_present == '1':
        
        #if eruptions are present, start with an AGN baseline
        #for QPEs time binning of 250s is acceptable, and durations of 120ks (although they may be truncated during feature extraction)
        times = np.arange(0,481*250,250,dtype=float)
        
        #draw a variety of different values for the underlying TK95 power law. These values are following the 
        #distribution identified by Gonzalez-Martin
        pl_indices = 2.06 + 0.01*np.random.randn(no_lcs)
        
        #simulate a series of Emmanoulopoulos lightcurves
        fluxes = np.zeros((no_lcs,481))
        for i in range(no_lcs):
            fluxes[i,:] = ELC_model()
        
        #draw a random eruption profile
        eruptions_profile = QPEProfile(length=481,amplitude=,width=,dutycycle=)
        
        #convolve the eruptions with the baseline to create the QPE lightcurve
        rate *= eruptions_profile
        
        #add poissonian noise to the result
        
        
        #discretise the result
        
        
        #create the output file location and output array
        


#output the file to the desired location
#np.savetxt(outfile_location,out_array,delimiter=',')

#plot the first five from the sample in order to check reasonableness
for i in range(5):
    plt.plot(out_array[0],out_array[i+1])
    plt.show()