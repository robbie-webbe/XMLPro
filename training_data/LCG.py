#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:15:45 2020

@author: do19150
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from numpy.fft import rfft, irfft
from scipy.stats import rankdata


def TKLC(T,dt,beta,break_freq=None,beta2=False,floor_freq=None,plot=False):
    '''
    Function for the creation of lightcurves as per the Timmer and Konig 1995 algorithm.
    
    Parameters:
        
    Returns:
        
    '''
    
    #create an output time array
    times = np.arange(0,T+dt,dt)
    #create an output frequency array
    freqs = np.fft.rfftfreq(n=int(T/dt),d=dt)
    
    #create an empty complex array for the psd
    power = np.zeros(len(freqs),dtype=complex)
    
    #populate the psd with randomly determined complex values
    power.real = np.random.randn(len(power))
    power.imag = np.random.randn(len(power))
    
    #set the zero frequency bin to power 0
    power[0] = 0+0j
    
    #if an even number of data points then set the power at the nyquist frequency to be real
    if len(times) % 2 == 0:
        power.imag[-1] = 0
        
    #multiply the psd by the power law with index beta
    power[1:] *= freqs[1:]**(-0.5*beta)

    #if there is to be a break in the power law at a given frequency
    if break_freq != None:
        #find the frequencies at which to alter the power for the different index
        break_indices = np.where(freqs >= break_freq)[0]
        #implement the new power las above that break frequency
        power[break_indices] *= ((1/break_freq)*freqs[break_indices])**(-0.5*(beta2-beta))
    
    #if there is to be a flat floor to the PSD at a given frequency
    if floor_freq != None:
        #find the frequencies at which to implement a floor
        floor_indices = np.where(freqs >= floor_freq)[0]
        #multiply those frequencies such that a floor is reached at that level
        power[floor_indices] *= (floor_freq**(-0.5*beta2))*freqs[floor_indices]**(0.5*beta2)
        
    rate = np.fft.irfft(power,len(times))
    
    if plot:
        plt.plot(times, rate)
        plt.show()    
        plt.loglog(freqs[1:],abs(power[1:])**2)
        plt.show()
    
    return times, rate, freqs, power


def ELC_model(lc_length,t_bin,TK95_beta,PDF_str,PDF_args=None,TK95_args=None,max_iterations=1e3,plot=False,verbose=False):
    '''
    Function for the creation of lightcurves as per the methodology outlined in Emmanoulopoulos et al,
    MNRAS, 433, 907-927. The function is designed to work as an extension to that prescribed by 
    Timmer & Konig in 1995. This function is designed to create lightcurves as per an underlying
    model describing its probability density function, rather than matching to known observations.
    
    Parameters:
        lc_length - Length of lightcurve to be generated, in time units, not the number of bins.
        t_bin - Time binning of lightcurve to be generated.
        TK95_beta - The index for the power law which will be used in creating the TK95 lightcurve and PSD.
        PDF_str - The underlying probability density function for the lightcurve. A string corresponding to a
                scipy.stats distribution.
        PDF_args - A dictionary of arguments for the prescribed scipy.stats distribution.
        converge_tol - Tolerance to which convergence must be achieved during the iterations.
        plot - Choose whether to plot the output lightcurve at every step of the iteration
    '''
    
    
    #determine the number of bins and add one to the number of bins to account for the first and last points being inclusive
    no_bins = int((lc_length / t_bin) + 1)
    
    #produce an array of time stamps for the length and sampling required
    times = np.arange(0,lc_length+t_bin,t_bin)
    
    #Algorithm step (i)
    #produce a PSD and time series using the TK95 algorithm 
    if TK95_args == None:
        tk_lc_psd = TKLC(lc_length,t_bin,TK95_beta)
    else:
        tk_lc_psd = TKLC(lc_length,t_bin,TK95_beta,**TK95_args)
    #unpack the amplitudes of the DFT from the TK algorithm as the only part of that routine which is needed.
    #rates = tk_lc_psd[1]
    tk_amps = np.abs(tk_lc_psd[3])
    #tk_angs = np.angle(tk_lc_psd[3])
    
    
    #Algorithm step (ii)
    #produce a set of random numbers using the PDF,
    #first identify the appropriate PDF from the input
    try:
        PDF = getattr(scipy.stats,PDF_str)
    except:
        print("That is not a recognised scipy.stats distribution. Please check the spelling and formatting.")
        return print("An invalid distirbution was returned. Start again.")
    else:
        PDF = getattr(scipy.stats,PDF_str)
    
    if PDF_args == None:
        return print("You have not specified the arguments required for the PDF. Start again")
    
    #then create a set of random 
    pdf_vals = PDF.rvs(**PDF_args,size=no_bins)
    if plot:
        plt.plot(pdf_vals)
        plt.show()
        plt.hist(pdf_vals)
        plt.show()
    #and then the DFT of the same
    pdf_psd = rfft(pdf_vals)
    #unpack the psd into its amplitudes and phases
    #pdf_psd_amps = np.absolute(pdf_psd)
    pdf_psd_phas = np.angle(pdf_psd)
    
    
    #iterate over these steps until convergence
    k=0
    adjusted_pdf_ts = np.zeros(len(pdf_vals))
    while k <= max_iterations:

        
        #Algorithm step (iii)
        #set the amplitudes in the PDF psd to the amplitudes from the TK PSD
        adjusted_pdf_psd_amps = tk_amps
        
        #create the adjusted psd from the PDF using the new angles
        adjusted_pdf_psd = adjusted_pdf_psd_amps * (np.cos(pdf_psd_phas) + 1j*np.sin(pdf_psd_phas))
        
        #now perform the inverse DFT
        adjusted_pdf_ts = irfft(adjusted_pdf_psd,no_bins)
        
        
        #Algorithm step (iv)
        #now create a new time series based on the magnitudes of the points from the original PDF lightcurve,
        #but distributed temporally according to the IDFT of the adjusted PDF PSD.
        
        #replace the highest point in the new adjusted PDF lightcurve with the value of the highest point from
        #initial_pdf_vals, and do so for all points in the adjusted pdf lightcurve
        ranking = rankdata(adjusted_pdf_ts).astype(int)-1
        adjusted_pdf_ts = np.sort(pdf_vals)[ranking]
        
        #increase the iteration count by one
        k += 1
        
        if np.array_equal(pdf_vals,adjusted_pdf_ts):
            break
        
        if plot:
            plt.plot(times,adjusted_pdf_ts)
            plt.scatter(times,pdf_vals)
            plt.show()
        
        #Algorithm step (v)
        #if the time series have not converged then set the current iteration to be the reference, and create 
        #a new PSD, and derived amplitudes and phases from it for the next iteration
        pdf_vals = adjusted_pdf_ts
        pdf_psd = rfft(pdf_vals)
        #pdf_psd_amps = np.absolute(pdf_psd)
        pdf_psd_phas = np.angle(pdf_psd)
        
    
    
    if np.array_equal(adjusted_pdf_ts,pdf_vals) == False:
        return print("Convergence was never reached. Try a greater number of iterations or a lightcurve with fewer time bins.")
    else:
        if verbose:
            print('Lightcurve converged on model after '+str(k)+' iterations.')
    
    return times, adjusted_pdf_ts





