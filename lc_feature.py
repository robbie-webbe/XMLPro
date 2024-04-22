#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.stats as stat
from scipy.signal import lombscargle
from itertools import groupby
from astropy.stats import bayesian_blocks
from astropy.io import fits
from tqdm import tqdm
from stingray import Lightcurve


'''
Function associated with the generation of features from lightcurves for the purposes of lightcurve classification. 
Features have been selected as statistical measures to describe simulated lightcurves without requiring simulated 
errors on counts or count rates, but may be included for observational data. Feature sets include several features 
which may or may not be deselected during the training of individual classifiers.

Features:
 - Proportion of points above mean - (5/4/3/2/1)*std
 - Proportion of points above mean + (1/2/3/4/5)*std
 - Proportion of points within +/- 1/2/3/4/5 std of the mean
 - IQR / standard deviation
 - standard deviation / Range
 - IQR / Range
 - Proportional position of the LQ, median, and UQ within the range
 - (LQ/Median/UQ - mean) / std 
 - Maximum absolute difference in consecutive points / std
 - Kurtosis
 - Skew
 - Robust Median Statistic (Sokolovsky 2017)
 - Median Absolute Deviation / std
 - Reverse Cross-Correlation normalised by the std
 - Autocorrelation peak beyond first zero
 - Consecutive Same-sign Deviation proportion (consecutive 3 points) (Sokolovsky 2017)
 - Consecutive Same-Sign Gradient proportion (consecutive 3 gradients , 4 points)
 - Lomb-Scargle Periodogram peak
 - Reduced Chi^2 against a constant value using the standard deviation
 - Normalised excess variance
 - Number of peaks from Bayesian block analysis (Scargle 1998) / number of data points
 - Amplitude of largest flare (requiring 3 consecutive points above mean + 2std) / std
 - Anderson-Darling Test
 - S_B variability detection statistic (Sokolovsky 2017)
 - Von Neumann Ratio (Sokolovsky 2017)
 - Excess Abbe value (Sokolovsky 2017)

'''

def simlc_feats(times, time_series, dt):
    """
    Function for the creation of the above feature set for a given time series data set.
    Args:
        times: List of time stamps for the time series
        time_series: Series of values, not necessarily evenly spaced, which describes the values of the time series.
        dt: Time binning of input time series

    Returns:
        features: Features derived from the lightcurve to be used in profiling.
    """

    # create an empty features array
    features = np.zeros(42)

    # determine the number of points, mean, median, IQR, and standard deviation for use in calculating future features.
    N = len(time_series)
    mean = np.average(time_series)
    median = np.median(time_series)
    std = np.std(time_series)
    iqr = stat.iqr(time_series)
    low = np.min(time_series)
    rng = np.max(time_series) - low
    lq = np.quantile(time_series, 0.25)
    uq = np.quantile(time_series, 0.75)


    # create a time series which is normalised by the mean and standard deviation
    norm_ts = (time_series - mean) / std

    # create the first ten features related to proportions of points -5 -> +5 std above the mean
    for i in range(5):
        features[i] = len(np.where(norm_ts >= (i-5))[0]) / N
        features[i+5] = len(np.where(norm_ts >= (i+1))[0]) / N

    # create features related to proportions of points with n stds of the mean.
    for i in range(5):
        features[i+10] = features[4-i] - features[i+5]

    # determine values of features related to IQR and quartiles normalised by the std
    features[15] = iqr / std
    features[16] = std / rng
    features[17] = iqr / rng

    # determine features related to the positions of the median and quartiles within the range
    features[18] = (lq - low) / rng
    features[19] = (median - low) / rng
    features[20] = (uq - low) / rng

    # determine features related to the difference between mean and quartiles as per the std
    features[21] = (lq - mean) / std
    features[22] = (median - mean) / std
    features[23] = (uq - mean) / std

    # determine the differences between consecutive points, and thence the maximum difference
    deviations = norm_ts[1:] - norm_ts[:-1]
    features[24] = np.max(np.abs(deviations))

    # calculate the kurtosis and skew of the lightcurve
    features[25] = stat.skew(time_series)
    features[26] = stat.kurtosis(time_series)

    # calculate the RoMS as per Sokolovsky and MAD
    features[27] = (1/(N-1))*np.sum(np.abs((time_series-median)/std))
    features[28] = stat.median_abs_deviation(norm_ts)

    # determine the reverse cross correlation and auto correlation-based features
    max_time = times[-1]
    rev_times = np.flip(max_time - times)
    rev_norm_ts = np.flip(norm_ts)
    for i in range(N):
        if times[i] in rev_times:
            features[29] += norm_ts[i]*rev_norm_ts[np.searchsorted(rev_times, times[i])]

    # determine the maximum lag that can be seen in the lightcurve
    max_lag = times[-1] - times[0]
    lag_vals = np.arange(0, max_lag+dt, dt)
    # create empty autocorrelation and pairs arrays
    ac_vals = np.zeros(len(lag_vals))
    ac_pairs = np.zeros(len(lag_vals))
    # iterate over lag lengths
    for i in range(len(lag_vals)):
        # for each time value, check if the corresponding lagged time exists, and
        # compute the ac if so
        for j in range(N):
            lag_time = times[j] + lag_vals[i]
            if lag_time in times:
                ac_vals[i] += norm_ts[j] * norm_ts[np.searchsorted(times, lag_time)]
                ac_pairs[i] += 1
    # make a nominal adjustment to the autocorrelation for the uneven sampling
    # leading to uneven numbers of pairs contributing to the autocorrelation
    ac = (ac_vals[1:]/ac_vals[0]) * np.flip(np.arange(1, len(lag_vals)))/ac_pairs[1:]
    ac_first0_idx = np.where(ac <= 0)[0][0]
    features[30] = np.max(ac[ac_first0_idx:])

    # determine the features dependent on consecutive sets of points, n can be
    # adjusted later if desired
    css_n = 3
    for i in range(N+1-css_n):
        if (norm_ts[i] >=0 and norm_ts[i+1] >=0 and norm_ts[i+2] >=0) or (norm_ts[i] <=0 and norm_ts[i+1] <=0 and norm_ts[i+2] <= 0):
            features[31] += 1
    features[31] /= (N+1-css_n)
    for i in range(N-css_n):
        if (norm_ts[i] <= norm_ts[i+1] <= norm_ts[i+2] <= norm_ts[i+3]) or (norm_ts[i] >= norm_ts[i+1] >= norm_ts[i+2] >= norm_ts[i+3]):
            features[32] += 1
    features[32] /= len(deviations)-(css_n-1)

    # create an output frequency range for the LS periodogram
    df = 1 / (5 * times[-1])
    ls_freqs = np.arange(start=df, stop=(10/times[1] + df), step=df)
    lsp = lombscargle(times, time_series, ls_freqs, normalize=True)
    features[33] = np.max(lsp)

    # determine the reduced chi squared against a constant value and excess variance
    features[34] = np.sum(norm_ts**2) / (N-1)
    features[35] = (1/(N*mean**2)) * np.sum((time_series-mean)**2-std**2)

    # create bayesian block bin edges for the time series
    bb = bayesian_blocks(times,time_series,fitness='measures')
    # rebin the bayesian blocks into continuous groups above and below a threshold of
    # mean and 3 * std, to determine sizes of individual flares.
    flare_block_times = []
    flare_block_counts = []
    no_flares = 0
    flare_block_times.append(bb[0])
    first_block_idxs = np.where((times >= bb[0]) & (times < bb[1]))[0]
    current_block_counts = norm_ts[first_block_idxs]
    if any(norm_ts[first_block_idxs] > 3):
        flare_state = 1
        no_flares += 1
    else:
        flare_state = 0
    for i in range(len(bb) - 2):
        # check if the next block contains points above 3 std
        block_idxs = np.where((times >= bb[i+1]) & (times < bb[i+2]))[0]
        # if it does then set the flare state to true
        if any(norm_ts[block_idxs] > 3):
            new_flare_state = 1
        else:
            new_flare_state = 0
        # if the new flare state matches the old one then move on to the next box
        if new_flare_state == flare_state:
            current_block_counts = np.concatenate((current_block_counts,norm_ts[block_idxs]))
            continue
        # if it doesn't, then add the box end to the blocks list, and change the flare state
        else:
            flare_block_counts.append(current_block_counts)
            current_block_counts = norm_ts[block_idxs]
            flare_state = new_flare_state
            flare_block_times.append(bb[i + 2])
            if new_flare_state == 1:
                no_flares += 1
    flare_block_times.append(bb[-1])
    flare_block_counts.append(current_block_counts)

    # determine the number of flares, and so the number of flares per lc length
    features[36] = no_flares / N

    # for each flare section, if they have at least three points above mean + 2*std, record the max
    for i in range(len(flare_block_counts)):
        if len(np.where(flare_block_counts[i] > 2)[0]) >= 3:
            features[37] = max(features[37],np.max(flare_block_counts[i]))

    # determine the Anderson-Darling probability
    features[38] = stat.anderson(time_series).statistic

    # for the s_b variability metric, first split the lightcurve into sets of same sign deviations from the mean
    sets = []
    for _, g in groupby(norm_ts, lambda x: x < 0):
        sets.append(list(g))
    # then calculate the metric based on the values and uncertainties
    residual_sums = []
    for i in range(len(sets)):
        residual_sums.append(np.sum(sets[i])**2)
    features[39] = (1 / (N*len(residual_sums))) * np.sum(residual_sums)

    # calculate the von Neumann ratio
    features[40] = (np.sum(deviations**2)/(N-1)) / std**2

    # define a time interval for calculating the Excess Abbe values. We choose 10*dt as an intermediate value dependent
    # on the original binning. Then iterate over all time series segments to calculate Abbe values for the intervals.
    interval_values = []
    for i in range(N):
        interval_start = times[i]
        interval_end = interval_start + 10 * dt
        if interval_end > times[-1]:
            continue
        interval_ts = time_series[np.where((times >= interval_start) & (times <= interval_end))[0]]
        interval_mean = np.average(interval_ts)
        interval_total_var = np.sum((interval_ts - interval_mean)**2)
        if interval_total_var == 0:
            continue
        interval_values.append((np.sum((interval_ts[1:] - interval_ts[:-1])**2)) / interval_total_var)
    # calculate the average value, and therefore the excess Abbe
    features[41] = np.average(interval_values) - features[40]/2

    return features


def obslc_feats(times, time_series, ts_err, dt):
    """
    Function for the creation of the above feature set for a given time series data set.
    Args:
        times: List of time stamps for the time series
        time_series: Series of values, not necessarily evenly spaced, which describes the values of the time series.
        ts_err: Errors on the measurements for time series
        dt: Time binning of input time series

    Returns:
        features: Features derived from the lightcurve to be used in profiling.
    """

    # create an empty features array
    features = np.zeros(42)

    # determine the number of points, mean, median, IQR, and standard deviation for use in calculating future features.
    N = len(time_series)
    mean = np.average(time_series)
    median = np.median(time_series)
    std = np.std(time_series)
    iqr = stat.iqr(time_series)
    low = np.min(time_series)
    rng = np.max(time_series) - low
    lq = np.quantile(time_series, 0.25)
    uq = np.quantile(time_series, 0.75)


    # create a time series which is normalised by the mean and errors on individual points
    norm_ts = (time_series - mean) / ts_err

    # create the first ten features related to proportions of points -5 -> +5 error above the mean
    for i in range(5):
        features[i] = len(np.where(norm_ts >= (i-5))[0]) / N
        features[i+5] = len(np.where(norm_ts >= (i+1))[0]) / N

    # create features related to proportions of points with n error of the mean.
    for i in range(5):
        features[i+10] = features[4-i] - features[i+5]

    # determine values of features related to IQR and quartiles normalised by the std
    features[15] = iqr / std
    features[16] = std / rng
    features[17] = iqr / rng

    # determine features related to the positions of the median and quartiles within the range
    features[18] = (lq - low) / rng
    features[19] = (median - low) / rng
    features[20] = (uq - low) / rng

    # determine features related to the difference between mean and quartiles as per the std
    features[21] = (lq - mean) / std
    features[22] = (median - mean) / std
    features[23] = (uq - mean) / std

    # determine the differences between consecutive points, and thence the maximum difference
    deviations = norm_ts[1:] - norm_ts[:-1]
    features[24] = np.max(np.abs(deviations))

    # calculate the kurtosis and skew of the lightcurve
    features[25] = stat.skew(time_series)
    features[26] = stat.kurtosis(time_series)

    # calculate the RoMS as per Sokolovsky and MAD
    features[27] = (1/(N-1))*np.sum(np.abs((time_series-median)/ts_err))
    features[28] = stat.median_abs_deviation(norm_ts)

    # determine the reverse cross correlation and auto correlation-based features
    max_time = times[-1]
    rev_times = np.flip(max_time - times)
    rev_norm_ts = np.flip(norm_ts)
    for i in range(N):
        if times[i] in rev_times:
            features[29] += norm_ts[i]*rev_norm_ts[np.searchsorted(rev_times, times[i])]

    # determine the maximum lag that can be seen in the lightcurve
    max_lag = times[-1] - times[0]
    lag_vals = np.arange(0, max_lag+dt, dt)
    # create empty autocorrelation and pairs arrays
    ac_vals = np.zeros(len(lag_vals))
    ac_pairs = np.zeros(len(lag_vals))
    # iterate over lag lengths
    for i in range(len(lag_vals)):
        # for each time value, check if the corresponding lagged time exists, and
        # compute the ac if so
        for j in range(N):
            lag_time = times[j] + lag_vals[i]
            if lag_time in times:
                ac_vals[i] += norm_ts[j] * norm_ts[np.searchsorted(times, lag_time)]
                ac_pairs[i] += 1
    # make a nominal adjustment to the autocorrelation for the uneven sampling
    # leading to uneven numbers of pairs contributing to the autocorrelation
    # if ac_pairs is 0 for any lag then mask out that bin.
    good_ac_idxs = np.where(ac_pairs[1:] != 0)
    ac = (ac_vals[1:]/ac_vals[0]) * np.flip(np.arange(1, len(lag_vals)))/ac_pairs[1:]
    ac = ac[good_ac_idxs]
    ac_first0_idx = np.where(ac <= 0)[0][0]
    features[30] = np.max(ac[ac_first0_idx:])

    # determine the features dependent on consecutive sets of points, n can be
    # adjusted later if desired
    css_n = 3
    for i in range(N+1-css_n):
        if (norm_ts[i] >=0 and norm_ts[i+1] >=0 and norm_ts[i+2] >=0) or (norm_ts[i] <=0 and norm_ts[i+1] <=0 and norm_ts[i+2] <= 0):
            features[31] += 1
    features[31] /= (N+1-css_n)
    for i in range(N-css_n):
        if (norm_ts[i] <= norm_ts[i+1] <= norm_ts[i+2] <= norm_ts[i+3]) or (norm_ts[i] >= norm_ts[i+1] >= norm_ts[i+2] >= norm_ts[i+3]):
            features[32] += 1
    features[32] /= len(deviations)-(css_n-1)

    # create an output frequency range for the LS periodogram
    df = 1 / (5 * times[-1])
    ls_freqs = np.arange(start=df, stop=(10/times[1] + df), step=df)
    lsp = lombscargle(times, time_series, ls_freqs, normalize=True)
    features[33] = np.max(lsp)

    # determine the reduced chi squared against a constant value and excess variance
    features[34] = np.sum(((time_series - mean)/ts_err)**2) / (N-1)
    features[35] = (1/(N*mean**2)) * np.sum((time_series-mean)**2-ts_err**2)

    # create bayesian block bin edges for the time series
    bb = bayesian_blocks(times,time_series,sigma=ts_err,fitness='measures')
    # rebin the bayesian blocks into continuous groups above and below a threshold of
    # mean and 3 * std, to determine sizes of individual flares.
    flare_block_times = []
    flare_block_counts = []
    no_flares = 0
    flare_block_times.append(bb[0])
    first_block_idxs = np.where((times >= bb[0]) & (times < bb[1]))[0]
    current_block_counts = norm_ts[first_block_idxs]
    if any(norm_ts[first_block_idxs] > 3):
        flare_state = 1
        no_flares += 1
    else:
        flare_state = 0
    for i in range(len(bb) - 2):
        # check if the next block contains points above 3 std
        block_idxs = np.where((times >= bb[i+1]) & (times < bb[i+2]))[0]
        # if it does then set the flare state to true
        if any(norm_ts[block_idxs] > 3):
            new_flare_state = 1
        else:
            new_flare_state = 0
        # if the new flare state matches the old one then move on to the next box
        if new_flare_state == flare_state:
            current_block_counts = np.concatenate((current_block_counts,norm_ts[block_idxs]))
            continue
        # if it doesn't, then add the box end to the blocks list, and change the flare state
        else:
            flare_block_counts.append(current_block_counts)
            current_block_counts = norm_ts[block_idxs]
            flare_state = new_flare_state
            flare_block_times.append(bb[i + 2])
            if new_flare_state == 1:
                no_flares += 1
    flare_block_times.append(bb[-1])
    flare_block_counts.append(current_block_counts)

    # determine the number of flares, and so the number of flares per lc length
    features[36] = no_flares / N

    # for each flare section, if they have at least three points above mean + 2*std, record the max
    for i in range(len(flare_block_counts)):
        if len(np.where(flare_block_counts[i] > 2)[0]) >= 3:
            features[37] = max(features[37],np.max(flare_block_counts[i]))

    # determine the Anderson-Darling probability
    features[38] = stat.anderson(time_series).statistic

    # for the s_b variability metric, first split the lightcurve into sets of same sign deviations from the mean
    sets = []
    err_norm_ts = (time_series - mean) / ts_err
    for _, g in groupby(err_norm_ts, lambda x: x < 0):
        sets.append(list(g))
    # then calculate the metric based on the values and uncertainties
    residual_sums = []
    for i in range(len(sets)):
        residual_sums.append(np.sum(sets[i])**2)
    features[39] = (1 / (N*len(residual_sums))) * np.sum(residual_sums)

    # calculate the von Neumann ratio
    features[40] = (np.sum(deviations**2)/(N-1)) / std**2

    # define a time interval for calculating the Excess Abbe values. We choose 10*dt as an intermediate value dependent
    # on the original binning. Then iterate over all time series segments to calculate Abbe values for the intervals.
    interval_values = []
    for i in range(N):
        interval_start = times[i]
        interval_end = interval_start + 10 * dt
        if interval_end > times[-1]:
            continue
        interval_ts = time_series[np.where((times >= interval_start) & (times <= interval_end))[0]]
        interval_mean = np.average(interval_ts)
        interval_total_var = np.sum((interval_ts - interval_mean)**2)
        if interval_total_var == 0:
            continue
        interval_values.append((np.sum((interval_ts[1:] - interval_ts[:-1])**2)) / interval_total_var)
    # calculate the average value, and therefore the excess Abbe
    features[41] = np.average(interval_values) - features[40]/2

    return features


def create_simobs_feats(filename):
    '''
    Creates the time series features vector for a given observation file, and outputs to a specified location.
    Args:
        filename: Filename for the given set of simulated lightcurves
    '''

    # Pick out the time binning from the sim lc filename
    dt = float(filename.split('/')[-1].split('_')[-1][2:-4])

    # load in the simulated lightcurves and time stamps
    data = np.loadtxt(infile, delimiter=',')

    # create an empty feature array and then loop all over simulated lcs
    feature_array = np.zeros((len(data) - 1, 42))
    for i in tqdm(range(len(data) - 1)):
        feature_array[i] = simlc_feats(data[0], data[i + 1], dt)

    # save the features to the outfile location
    np.savetxt('feature_extraction/' + infile.split('/')[-1][:-4] + '_feats.csv', feature_array, delimiter=',')


def create_xmmobs_feats(filename, delt):
    '''
    Creates the time series features vector for a given observation file. Observation file should be in the
    standard XMM pps format.

    Args:
        filename: Filename for the given XMM observation and source number
        delt: Time binning required for the lightcurve for the phenomena of interest

    Returns:
        full_feats: Features for the full band lightcurve for energies from 0.2-12.0 keV
        soft_feats: Features for the soft band lightcurve for energies from 0.2-2.0 keV
        hard_feats: Features for the hard band lightcurve for energies from 2.0-12.0 keV
    '''

    # open the pps time series file
    hdul = fits.open(filename)

    # pick out the time stamps, intrinsic dt, rates in each band, and rate errors
    time = hdul['RATE'].data.field('TIME')
    t0 = time[0]
    time -= t0
    dt = hdul['RATE'].data.field('TIMEDEL')
    full_count = hdul['RATE'].data.field('RATE') * dt
    full_err = hdul['RATE'].data.field('ERROR') * dt
    soft_count = (hdul['RATE'].data.field('RATE1') + hdul['RATE'].data.field('RATE2') + hdul['RATE'].data.field('RATE3')) * dt
    soft_err = (hdul['RATE'].data.field('RATE1_ERR') + hdul['RATE'].data.field('RATE2_ERR') + hdul['RATE'].data.field('RATE3_ERR')) * dt
    hard_count = (hdul['RATE'].data.field('RATE4') + hdul['RATE'].data.field('RATE5')) * dt
    hard_err = (hdul['RATE'].data.field('RATE4_ERR') + hdul['RATE'].data.field('RATE5_ERR')) * dt

    # create the gti information in a stingray related format
    gtis = []
    for i in hdul['SRC_GTIS'].data:
        gtis.append([i[0] - t0, i[1] - t0])

    #create the lightcurve objects for each band where the rate and errors are finite
    full_lc_mask = np.where(np.isfinite(full_count) & np.isfinite(full_err))
    soft_lc_mask = np.where(np.isfinite(soft_count) & np.isfinite(soft_err))
    hard_lc_mask = np.where(np.isfinite(hard_count) & np.isfinite(hard_err))
    full_lc = Lightcurve(time[full_lc_mask], full_count[full_lc_mask], err=full_err[full_lc_mask])
    soft_lc = Lightcurve(time[soft_lc_mask], soft_count[soft_lc_mask], err=soft_err[soft_lc_mask])
    hard_lc = Lightcurve(time[hard_lc_mask], hard_count[hard_lc_mask], err=hard_err[hard_lc_mask])

    # rebin to the required rate, and include GTI data
    full_lc = full_lc.rebin(delt)
    soft_lc = soft_lc.rebin(delt)
    hard_lc = hard_lc.rebin(delt)
    full_lc.gti = gtis
    soft_lc.gti = gtis
    hard_lc.gti = gtis
    full_lc.apply_gtis()
    soft_lc.apply_gtis()
    hard_lc.apply_gtis()

    # if there are any points where the errors are zero, make them infinite instead
    full_lc.counts_err[np.where(full_lc.counts_err == 0)[0]] = np.inf
    soft_lc.counts_err[np.where(soft_lc.counts_err == 0)[0]] = np.inf
    hard_lc.counts_err[np.where(hard_lc.counts_err == 0)[0]] = np.inf

    # create the feature arrays for the three lightcurves
    full_feats = obslc_feats(full_lc.time, full_lc.counts, full_lc.counts_err, delt)
    soft_feats = obslc_feats(soft_lc.time, soft_lc.counts, soft_lc.counts_err, delt)
    hard_feats = obslc_feats(hard_lc.time, hard_lc.counts, hard_lc.counts_err, delt)

    return full_feats, soft_feats, hard_feats
