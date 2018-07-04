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
    resultList = []    
    resultList.append(0)
    for i in range(len(toCalcList)-1):
        resultList.append((toCalcList[i+1] - toCalcList[i])/(timestampList[i+1] - timestampList[i]))
    return resultList

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
    data_watch[i]['x_acc'] = (data_watch[i]['x'] - data_watch[i]['x'].mean())
    data_watch[i]['y_acc'] = (data_watch[i]['y'] - data_watch[i]['y'].mean())
    data_watch[i]['z_acc'] = (data_watch[i]['z'] - data_watch[i]['z'].mean())
    data_watch[i]['filtered_x_acc'] = sig.savgol_filter(data_watch[i]['x_acc'], 31, 5)
    data_watch[i]['filtered_y_acc'] = sig.savgol_filter(data_watch[i]['y_acc'], 31, 5)
    data_watch[i]['filtered_z_acc'] = sig.savgol_filter(data_watch[i]['z_acc'], 31, 5)
    
    a = np.column_stack((data_watch[i]['x_acc'], data_watch[i]['y_acc'], data_watch[i]['z_acc']))
    b = np.zeros(a.shape[0])
    c = np.zeros(a.shape[0])
    
    for j in range(a.shape[0]):
        b[j] = np.linalg.norm(a[j]) # Euclidean norm = 2-norm -> for calculating the non-gravitational norm
    
    data_watch[i]['ngmagnitude'] = b
    data_watch[i]['filtered_ngmagnitude'] = sig.savgol_filter(b, 31, 5)
    
    data_watch[i]['x_vel'] = data_watch[i]['x_acc'].cumsum()
    data_watch[i]['y_vel'] = data_watch[i]['y_acc'].cumsum()
    data_watch[i]['z_vel'] = data_watch[i]['z_acc'].cumsum()
    
    data_watch[i]['x_pos'] = data_watch[i]['x_vel'].cumsum()
    data_watch[i]['y_pos'] = data_watch[i]['y_vel'].cumsum()
    data_watch[i]['z_pos'] = data_watch[i]['z_vel'].cumsum()
    
    smoothed_zscore = sz.thresholding_algo(data_watch[i]['filtered_ngmagnitude'], 5, 3, 0) # thresholding_algo(y, lag, threshold, influence)
    data_watch[i]['filtered_ngmagnitude_peaks'] = smoothed_zscore.get('signals')
    firstpeak_index = findFirstPeak(np.array(data_watch[i]['filtered_ngmagnitude_peaks']))
    peaks_difference = data_watch[i]['timestamp'][firstpeak_index] - data_phone[i]['timestamp'][0]    
    data_watch[i]['syncronised_timestamp'] = (data_watch[i]['timestamp'] - peaks_difference)
    
    diff = data_phone[i]['timestamp'][len(data_phone[i]['timestamp'])-1] - data_phone[i]['timestamp'][0]
    print(diff)
    approx_final_watch_timestamp = data_watch[i]['syncronised_timestamp'][firstpeak_index] + diff
    print(approx_final_watch_timestamp)
    final_watch_timestamp = find_nearest(data_watch[i]['syncronised_timestamp'].values, approx_final_watch_timestamp)
    print(final_watch_timestamp)
    
    final_index = pd.Index(data_watch[i]['syncronised_timestamp']).get_loc(final_watch_timestamp)
    print(final_index)
    data_watch[i]['syncronised_timestamp'] = data_watch[i]['syncronised_timestamp'].loc[firstpeak_index:final_index]
    
for i in range(len(files_phone)):
    fig, ax = plt.subplots(1, 1)
    data_phone[i][['x', 'y']] = data_phone[i][['x', 'y']] / 1000.0
    data_phone[i][['x-velocity', 'y-velocity']] = data_phone[i][['x-velocity', 'y-velocity']] / 20.0
    x_vel = np.array(data_phone[i]['x-velocity'])
    y_vel = np.array(data_phone[i]['y-velocity'])
    
    data_phone[i]['x-acc'] = calculateDerivativeList(data_phone[i]['timestamp'],x_vel)
    data_phone[i]['y-acc'] = calculateDerivativeList(data_phone[i]['timestamp'],y_vel) 
    data_phone[i]['filtered_x-acc'] = sig.savgol_filter(data_phone[i]['x-acc'], 31, 5)
    data_phone[i]['filtered_y-acc'] = sig.savgol_filter(data_phone[i]['y-acc'], 31, 5)
    data_phone[i][[
            'timestamp',
#            'x',
#            'y',
#            'x-velocity',
#            'y-velocity',
            'filtered_x-acc',
#            'filtered_y-acc'
            ]].plot(ax=ax, x='timestamp')
    
    data_watch[i][['x_vel', 'y_vel']] = data_watch[i][['x_vel', 'y_vel']] / 20.0

    data_watch[i][[
#             'timestamp',
              'syncronised_timestamp',  
             'x_acc', 
#             'y_acc', 
#             'z_acc',
#             'filtered_x_acc', 
#             'filtered_y_acc', 
#             'filtered_z_acc',
#             'ngmagnitude',
#             'filtered_ngmagnitude',
#             'filtered_ngmagnitude_peaks',
#            'x_vel', 
#            'y_vel', 
#            'z_vel', 
#            'x_pos', 
#            'y_pos', 
#            'z_pos'
            ]].plot(ax=ax, x='syncronised_timestamp', linestyle=':')
    plt.title(files_phone[i])


    