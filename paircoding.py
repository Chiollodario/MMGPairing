# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:41:47 2018

@author: DARIO-DELL
"""

import pandas as pd
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import smoothed_zscore as sz
import glob

plt.close('all')

# DataFrame collection from files
files_phone = glob.glob('C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-06-28_16_smartphone_sample.csv')
files_watch = glob.glob('C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-06-28_16_watch_sample.csv')

data_phone = [pd.read_csv(x) for x in files_phone] # List comprehension
data_watch = [pd.read_csv(x) for x in files_watch]


# derivative calculated as: Δx/Δt
def calculateDerivativeList(timestampList , toCalcList):
    toCalc = np.asarray(toCalcList)
    resultList = []    
    resultList.append(0)
    for i in range(len(toCalc)-1):
        resultList.append((toCalc[i+1] - toCalc[i])/(timestampList[i+1] - timestampList[i]))
    return resultList

# function for first peak detection of a signal
def findFirstPeak(a):
    i = 0
    while(a[i]!=1 and i<len(a)):
        i += 1
    return i

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


for i in range(len(files_watch)):
    # calculate the linear acceleration for each axis (gravity removal)    
    data_watch[i]['linear_x_acc'] = (data_watch[i]['x'] - data_watch[i]['x'].mean())
    data_watch[i]['linear_y_acc'] = (data_watch[i]['y'] - data_watch[i]['y'].mean())
    data_watch[i]['linear_z_acc'] = (data_watch[i]['z'] - data_watch[i]['z'].mean())
    
    # linear acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_linear_x_acc'] = sig.savgol_filter(data_watch[i]['linear_x_acc'], 31, 5)
    data_watch[i]['filtered_linear_y_acc'] = sig.savgol_filter(data_watch[i]['linear_y_acc'], 31, 5)
    data_watch[i]['filtered_linear_z_acc'] = sig.savgol_filter(data_watch[i]['linear_z_acc'], 31, 5)
    
    # calculate the Euclidean norm = 2-norm (that is: the magnitude)
    a = np.column_stack((data_watch[i]['linear_x_acc'], data_watch[i]['linear_y_acc'], data_watch[i]['linear_z_acc']))
    b = np.zeros(a.shape[0])
    for j in range(a.shape[0]):
        b[j] = np.linalg.norm(a[j]) # Euclidean norm = 2-norm 
    data_watch[i]['magnitude'] = b
    
    # magnitude noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_magnitude'] = sig.savgol_filter(b, 31, 5)
    
    # calculate the velocity (as the integral of the linear acceleration)
    data_watch[i]['x_vel'] = data_watch[i]['linear_x_acc'].cumsum()
    data_watch[i]['y_vel'] = data_watch[i]['linear_y_acc'].cumsum()
    data_watch[i]['z_vel'] = data_watch[i]['linear_z_acc'].cumsum()
    
    # scale x_vel, y_vel position 
    data_watch[i][['x_vel', 'y_vel']] = data_watch[i][['x_vel', 'y_vel']] / 20.0
    
    # velocity noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_x_vel'] = sig.savgol_filter(b, 31, 5)
    data_watch[i]['filtered_y_vel'] = sig.savgol_filter(b, 31, 5)
    data_watch[i]['filtered_z_vel'] = sig.savgol_filter(b, 31, 5)
    
    # calculate the position (as the integral of the velocity)
    data_watch[i]['x_pos'] = data_watch[i]['x_vel'].cumsum()
    data_watch[i]['y_pos'] = data_watch[i]['y_vel'].cumsum()
    data_watch[i]['z_pos'] = data_watch[i]['z_vel'].cumsum()
    
    # position noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_x_pos'] = sig.savgol_filter(b, 31, 5)
    data_watch[i]['filtered_y_pos'] = sig.savgol_filter(b, 31, 5)
    data_watch[i]['filtered_z_pos'] = sig.savgol_filter(b, 31, 5)
    
    # peak detection
    smoothed_zscore = sz.thresholding_algo(data_watch[i]['filtered_magnitude'], 5, 3, 0) # thresholding_algo(y, lag, threshold, influence)
    data_watch[i]['filtered_magnitude_peaks'] = smoothed_zscore.get('signals')
    
    # first peak detection for synchronising smartwatch and smartphone signals
    firstpeak_index = findFirstPeak(np.array(data_watch[i]['filtered_magnitude_peaks'])) # find out the very first peak
    peaks_difference = data_watch[i]['timestamp'][firstpeak_index] - data_phone[i]['timestamp'][0] # time difference of the two devices' timestamps   
    data_watch[i]['syncronised_timestamp'] = (data_watch[i]['timestamp'] - peaks_difference) # shift of the watch timestamps for synchronising the signals
    
    # detect the timestamp interval on the smartwatch (goal: discard useless datapoints from the smartwatch samples)
    diff = data_phone[i]['timestamp'][len(data_phone[i]['timestamp'])-1] - data_phone[i]['timestamp'][0] # drawing time on the smartphone screen
    approx_final_watch_timestamp = data_watch[i]['syncronised_timestamp'][firstpeak_index] + diff # approximate timestamp of the last meaningful smartwatch movement
    final_watch_timestamp = find_nearest(data_watch[i]['syncronised_timestamp'].values, approx_final_watch_timestamp) # actual timestamp of the last meaningful smartwatch movement
    final_index = pd.Index(data_watch[i]['syncronised_timestamp']).get_loc(final_watch_timestamp) # index of actual timestamp
    data_watch[i]['syncronised_timestamp'] = data_watch[i]['syncronised_timestamp'].loc[firstpeak_index:final_index] # discard of the non-meaningful datapoints in the sample
    
    
for i in range(len(files_phone)):
    fig, ax = plt.subplots(1, 1)
    
    # scale x, y position 
    data_phone[i][['x_pos', 'y_pos']] = data_phone[i][['x', 'y']] / 1000.0    
    # realign the y axis (smartphones have a different y-axis direction)
    data_phone[i]['y_pos'] = data_phone[i]['y_pos'] * -1
    
    # scale x, y velocity
    data_phone[i][['x_vel', 'y_vel']] = data_phone[i][['x-velocity', 'y-velocity']] / 20.0    
    # realign the y axis (smartphones have a different y-axis direction)
    data_phone[i]['y_vel'] = data_phone[i]['y_vel'] * -1
    
    # velocity noise filtering (through built-in Savitzky-Golay filter)
    data_phone[i]['filtered_x_vel'] = sig.savgol_filter(data_phone[i]['x_vel'], 31, 5)
    data_phone[i]['filtered_y_vel'] = sig.savgol_filter(data_phone[i]['y_vel'], 31, 5)
    
    # calculate the acceleration (as the first-derivative of the velocity)
    data_phone[i]['x_acc'] = calculateDerivativeList(data_phone[i]['timestamp'],data_phone[i]['x_vel'])
    data_phone[i]['y_acc'] = calculateDerivativeList(data_phone[i]['timestamp'],data_phone[i]['y_vel']) 
    
    # acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_phone[i]['filtered_x_acc'] = sig.savgol_filter(data_phone[i]['x_acc'], 31, 5)
    data_phone[i]['filtered_y_acc'] = sig.savgol_filter(data_phone[i]['y_acc'], 31, 5)
    
    
    data_phone[i][[
            'timestamp',
            
#            'x_pos', 
#            'y_pos',
            
#            'x_vel',
#            'y_vel',
            
#            'filtered_x_vel',
            'filtered_y_vel',
            
#            'x_acc',
#            'y_acc',
            
#            'filtered_x-acc',
#            'filtered_y-acc'
            ]].plot(ax=ax, x='timestamp')
    
    
    data_watch[i][[
#            'timestamp',
            'syncronised_timestamp',
    
#            'x_pos', 
#            'y_pos', 
#            'z_pos',
    
#            'filtered_x_pos',
#            'filtered_y_pos',
#            'filtered_z_pos',
                
#            'x_vel', 
#            'y_vel', 
#            'z_vel',
    
#            'filtered_x_vel',
            'filtered_y_vel',
#            'filtered_z_vel',
    
#            'linear_x_acc', 
#            'linear_y_acc', 
#            'linear_z_acc',
    
#            'filtered_linear_x_acc', 
#            'filtered_linear_y_acc', 
#            'filtered_linear_z_acc',
    
#            'magnitude',
#            'filtered_magnitude',
#            'filtered_magnitude_peaks',
            ]].plot(ax=ax, x='syncronised_timestamp', linestyle=':')
    plt.title(files_phone[i])


    