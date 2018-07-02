# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:41:47 2018

@author: DARIO-DELL
"""

import pandas as pd
import numpy as np
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


for i in range(len(files_watch)):    
    data_watch[i]['x_acc'] = (data_watch[i]['x'] - data_watch[i]['x'].mean())
    data_watch[i]['y_acc'] = (data_watch[i]['y'] - data_watch[i]['y'].mean())
    data_watch[i]['z_acc'] = (data_watch[i]['z'] - data_watch[i]['z'].mean())
    
    a = np.column_stack((data_watch[i]['x_acc'], data_watch[i]['y_acc'], data_watch[i]['z_acc']))
    b = np.zeros(a.shape[0])
    c = np.zeros(a.shape[0])
    
    for j in range(a.shape[0]):
        b[j] = np.linalg.norm(a[j]) # Euclidean norm = 2-norm
        c[j] = np.var(b[j])    
    data_watch[i]['ngmagnitude'] = b
    data_watch[i]['variance'] = c
    
    data_watch[i]['x_vel'] = data_watch[i]['x_acc'].cumsum()
    data_watch[i]['y_vel'] = data_watch[i]['y_acc'].cumsum()
    data_watch[i]['z_vel'] = data_watch[i]['z_acc'].cumsum()
    
    data_watch[i]['x_pos'] = data_watch[i]['x_vel'].cumsum()
    data_watch[i]['y_pos'] = data_watch[i]['y_vel'].cumsum()
    data_watch[i]['z_pos'] = data_watch[i]['z_vel'].cumsum()
    
    plt.figure(); 
    #data_watch[i]['ngmagnitude'].rolling(window=30, center=True).var().rolling(window=20, center=True).mean().plot()
    # thresholding_algo(y, lag, threshold, influence)
    result = sz.thresholding_algo(b, 30, 5, 0)
    
#for i in range(len(files_phone)):
#    fig, ax = plt.subplots(1, 1)
#    data_phone[i][['x', 'y']] = data_phone[i][['x', 'y']] / 50.0
#    x_vel = np.array(data_phone[i]['x-velocity'])
#    y_vel = np.array(data_phone[i]['y-velocity'])
##    x_vel, y_vel = data_phone[i][['x-velocity', 'y-velocity']] / 50.0
#    data_phone[i]['x-acc'] = calculateDerivativeList(data_phone[i]['timestamp'],x_vel)
#    data_phone[i]['y-acc'] = calculateDerivativeList(data_phone[i]['timestamp'],y_vel)    
#    
#    data_phone[i][['timestamp', 'x-acc']].plot(ax=ax, x='timestamp')
#    data_watch[i][[
#            'timestamp', 
#            'x_acc', 
##            'y_acc', 
##            'z_acc',
#             'ngmagnitude',
#             'variance',
##            'x_vel', 
##            'y_vel', 
##            'z_vel', 
##            'x_pos', 
##            'y_pos', 
##            'z_pos'
#            ]].plot(ax=ax, x='timestamp')
#    plt.title(files_phone[i])


    