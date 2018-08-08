import pandas as pd
import numpy as np
import scipy.signal as sig
#from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import smoothed_zscore as sz
import glob

plt.close('all')

# DataFrame collection from files
files_phone = glob.glob('C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-08-07_9_smartphone_sample.csv')
files_watch = glob.glob('C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-08-07_9_watch_sample.csv')

data_phone = [pd.read_csv(x) for x in files_phone] # List comprehension
data_watch = [pd.read_csv(x) for x in files_watch]

# utlity function for getting necessary info for the integral calculation
def f(x, key):
    index = pd.Index(data_watch[i]['timestamp']).get_loc(x)
    value = data_watch[i][key][index]
    return value

# integral calculated as the area beneath the graph, for each datapoint couple
def calculateIntegralList(timestampList, key):
    toCalc = np.array(timestampList.tolist())
    resultList = []
    resultList.append(0)
    for i in range(len(toCalc)):
        if i == len(toCalc)-1: 
            break
        y1 = f(toCalc[i], key)
        y2 = f(toCalc[i+1], key)
        integral = romb(np.array([y1,y2]), toCalc[i+1]-toCalc[i])
        resultList.append(integral)
    return resultList

# derivative calculated as: Δx/Δt
def calculateDerivativeList(timestampList, toCalcList):
    toCalc = np.asarray(toCalcList)
    resultList = []    
    resultList.append(0)
    for i in range(len(toCalc)):
        if i == len(toCalc)-1: 
            break
        resultList.append((toCalc[i+1] - toCalc[i])/(timestampList[i+1] - timestampList[i]))
    return resultList

# function for first peak detection of a signal
def findFirstPeak(a):
    i = 0
    while(i<len(a) and a[i]!=1):
        i += 1
    return i

# search for the closest value in a given array 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# === some commented parts are left for research purpose (in case they are needed for different tries)
# WATCH DATA ANALYSIS    
for i in range(len(files_watch)):
    
    #%% WATCH - PEAK DETECTION
    # 1st step: Savitzky–Golay filter - final rough accuracy = 90%
    # 2st step: Smoothed z-score algorithm - final rough accuracy = low since it returns a digital result
    
    # acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_x_acc'] = sig.savgol_filter(data_watch[i]['x'], 15, 5)
    data_watch[i]['filtered_y_acc'] = sig.savgol_filter(data_watch[i]['y'], 15, 5)
    data_watch[i]['filtered_z_acc'] = sig.savgol_filter(data_watch[i]['z'], 15, 5)
    
    # calculate the Euclidean norm = 2-norm (that is: the magnitude)
    a = np.column_stack((data_watch[i]['filtered_x_acc'], data_watch[i]['filtered_y_acc'], data_watch[i]['filtered_z_acc']))
    b = np.zeros(a.shape[0])
    for j in range(a.shape[0]):
        b[j] = np.linalg.norm(a[j]) # Euclidean norm = 2-norm 
    data_watch[i]['magnitude'] = b
    
    # magnitude noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_magnitude'] = sig.savgol_filter(b, 15, 5)    
    
    # peak detection
    # lag = the lag of the moving window 
    # threshold = the z-score at which the algorithm signals and influence
    # influence = (between 0 and 1) of new signals on the mean and standard deviation
    smoothed_zscore = sz.thresholding_algo(data_watch[i]['filtered_magnitude'], 55, 5, 0) # thresholding_algo(y, lag, threshold, influence)
    data_watch[i]['filtered_magnitude_peaks'] = smoothed_zscore.get('signals')
    
    #%% WATCH - SIGNAL SYNCHRONISATION
    
    # first peak detection for synchronising smartwatch and smartphone signals
    firstpeak_index = findFirstPeak(np.array(data_watch[i]['filtered_magnitude_peaks'])) # find out the very first peak
    peaks_difference = data_watch[i]['timestamp'][firstpeak_index] - data_phone[i]['timestamp'][0] # time difference of the two devices' timestamps   
    data_watch[i]['timestamp'] = data_watch[i]['timestamp'] - peaks_difference # shift of the watch timestamps for synchronising the signals
    
    # detect the timestamp interval on the smartwatch (goal: discard useless datapoints from the smartwatch samples)
    diff = data_phone[i]['timestamp'].iloc[-1] - data_phone[i]['timestamp'][0] # drawing time on the smartphone screen
    approx_final_watch_timestamp = data_watch[i]['timestamp'][firstpeak_index] + diff # approximate timestamp of the last meaningful smartwatch movement
    final_watch_timestamp = find_nearest(data_watch[i]['timestamp'].values, approx_final_watch_timestamp) # actual timestamp of the last meaningful smartwatch movement
    final_index = pd.Index(data_watch[i]['timestamp']).get_loc(final_watch_timestamp) # index of actual timestamp
    
    #%% WATCH - GRAVITY REMOVAL
    # 1st step: removal of the gravity mean for each of the axis only from the first to the final peak - final rough accuracy = 80% 
    # 2st step: Savitzky–Golay filter - final rough accuracy = 70%
    
    # calculate the linear acceleration for each axis (gravity removal in the interval firstpeak_index:final_index)
    temp_linear_x_acc = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_linear_y_acc = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_linear_z_acc = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_linear_x_acc[0:firstpeak_index] = data_watch[i]['x'][0:firstpeak_index] - data_watch[i]['x'][firstpeak_index:final_index+1].mean()
    temp_linear_y_acc[0:firstpeak_index] = data_watch[i]['y'][0:firstpeak_index] - data_watch[i]['y'][firstpeak_index:final_index+1].mean()
    temp_linear_z_acc[0:firstpeak_index] = data_watch[i]['z'][0:firstpeak_index] - data_watch[i]['z'][firstpeak_index:final_index+1].mean()
    temp_linear_x_acc[firstpeak_index:final_index+1] = data_watch[i]['x'][firstpeak_index:final_index+1] - data_watch[i]['x'][firstpeak_index:final_index+1].mean()
    temp_linear_y_acc[firstpeak_index:final_index+1] = data_watch[i]['y'][firstpeak_index:final_index+1] - data_watch[i]['y'][firstpeak_index:final_index+1].mean()
    temp_linear_z_acc[firstpeak_index:final_index+1] = data_watch[i]['z'][firstpeak_index:final_index+1] - data_watch[i]['z'][firstpeak_index:final_index+1].mean()
    temp_linear_x_acc[final_index+1:] = data_watch[i]['x'][final_index+1:] - data_watch[i]['x'][firstpeak_index:final_index+1].mean()
    temp_linear_y_acc[final_index+1:] = data_watch[i]['y'][final_index+1:] - data_watch[i]['y'][firstpeak_index:final_index+1].mean()
    temp_linear_z_acc[final_index+1:] = data_watch[i]['z'][final_index+1:] - data_watch[i]['z'][firstpeak_index:final_index+1].mean()
    data_watch[i]['linear_x_acc'] = temp_linear_x_acc
    data_watch[i]['linear_y_acc'] = temp_linear_y_acc
    data_watch[i]['linear_z_acc'] = temp_linear_z_acc
    
    # linear acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_linear_x_acc'] = sig.savgol_filter(data_watch[i]['linear_x_acc'], 15, 5)
    data_watch[i]['filtered_linear_y_acc'] = sig.savgol_filter(data_watch[i]['linear_y_acc'], 15, 5)
    data_watch[i]['filtered_linear_z_acc'] = sig.savgol_filter(data_watch[i]['linear_z_acc'], 15, 5)   
    
    #%% WATCH - VELOCITY CALCULATION
    # 1st step: "cumtrapz" integral calcolous starting from linear acceleration - final rough accuracy = 80%
    # 2st step: Savitzky–Golay filter - final rough accuracy = 70%
    
    # calculate velocity as the cumulative sum of the trapeziums beneath the crest
    temp_x_vel = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_y_vel = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_z_vel = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_x_vel[firstpeak_index:final_index+1] = cumtrapz(data_watch[i]['linear_x_acc'][firstpeak_index:final_index+1], data_watch[i]['timestamp'][firstpeak_index:final_index+1], initial=0)
    temp_y_vel[firstpeak_index:final_index+1] = cumtrapz(data_watch[i]['linear_y_acc'][firstpeak_index:final_index+1], data_watch[i]['timestamp'][firstpeak_index:final_index+1], initial=0)
    temp_z_vel[firstpeak_index:final_index+1] = cumtrapz(data_watch[i]['linear_z_acc'][firstpeak_index:final_index+1], data_watch[i]['timestamp'][firstpeak_index:final_index+1], initial=0)
    data_watch[i]['x_vel'] = temp_x_vel
    data_watch[i]['y_vel'] = temp_y_vel
    data_watch[i]['z_vel'] = temp_z_vel
    
    # velocity noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_x_vel'] = sig.savgol_filter(data_watch[i]['x_vel'], 15, 5)
    data_watch[i]['filtered_y_vel'] = sig.savgol_filter(data_watch[i]['y_vel'], 15, 5)
    data_watch[i]['filtered_z_vel'] = sig.savgol_filter(data_watch[i]['z_vel'], 15, 5)
    
    #%% WATCH - POSITION CALCULATION
    # 1st step: "cumtrapz" integral calcolous starting from velocity - final rough accuracy = 80%
    # 2st step: Savitzky–Golay filter - final rough accuracy = 70%
    
    # calculate the position (as the cumulative sum of the velocity)
    temp_x_pos = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_y_pos = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_z_pos = np.full(len(data_watch[i]['timestamp']),np.nan)
    temp_x_pos[firstpeak_index:final_index+1] = cumtrapz(data_watch[i]['x_vel'][firstpeak_index:final_index+1], data_watch[i]['timestamp'][firstpeak_index:final_index+1], initial=0)
    temp_y_pos[firstpeak_index:final_index+1] = cumtrapz(data_watch[i]['y_vel'][firstpeak_index:final_index+1], data_watch[i]['timestamp'][firstpeak_index:final_index+1], initial=0)
    temp_z_pos[firstpeak_index:final_index+1] = cumtrapz(data_watch[i]['z_vel'][firstpeak_index:final_index+1], data_watch[i]['timestamp'][firstpeak_index:final_index+1], initial=0)
    data_watch[i]['x_pos'] = temp_x_pos
    data_watch[i]['y_pos'] = temp_y_pos
    data_watch[i]['z_pos'] = temp_z_pos
    
    # position noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_x_pos'] = sig.savgol_filter(data_watch[i]['x_pos'], 15, 5)
    data_watch[i]['filtered_y_pos'] = sig.savgol_filter(data_watch[i]['y_pos'], 15, 5)
    data_watch[i]['filtered_z_pos'] = sig.savgol_filter(data_watch[i]['z_pos'], 15, 5)
    
# PHONE DATA ANALYSIS
for i in range(len(files_phone)):
    fig, ax = plt.subplots(1, 1)
    
    #%% PHONE - GET POSITION FROM API
    # rough accuracy = 100%
    
    # scale x, y position 
    data_phone[i][['x_pos', 'y_pos']] = data_phone[i][['x', 'y']]    
    # realign the y axis (smartphones have a different y-axis direction)
    data_phone[i]['y_pos'] = data_phone[i]['y_pos'] * -1

    #%% PHONE - GET VELOCITY FROM API
    # 1st step: rough accuracy = 100%
    # 2st step: Savitzky–Golay filter - final rough accuracy = 90%

#    # calculate the x,y velocity dervating the positions
#    data_phone[i]['x_vel'] = calculateDerivativeList(data_phone[i]['timestamp'], data_phone[i]['x_pos'])
#    data_phone[i]['y_vel'] = calculateDerivativeList(data_phone[i]['timestamp'], data_phone[i]['y_pos'])
     
    # get the x,y velocity from the MotionEvent API directly from the phone raw data
    data_phone[i][['x_vel', 'y_vel']] = data_phone[i][['x_velocity', 'y_velocity']]
    data_phone[i]['y_vel'] = data_phone[i]['y_vel'] * -1
    
    # scale x, y velocity
    data_phone[i][['x_vel', 'y_vel']] = data_phone[i][['x_vel', 'y_vel']] * 20.0

    # realign the y axis (smartphones have a different y-axis direction)
    data_phone[i]['y_vel'] = data_phone[i]['y_vel'] * -1
    
    # velocity noise filtering (through built-in Savitzky-Golay filter)
    data_phone[i]['filtered_x_vel'] = sig.savgol_filter(data_phone[i]['x_vel'], 15, 5)
    data_phone[i]['filtered_y_vel'] = sig.savgol_filter(data_phone[i]['y_vel'], 15, 5)
    
    #%% PHONE - ACCELERATION CALCULATION
    # 1st step: derivative calculation - rough accuracy = 100%
    # 2st step: Savitzky–Golay filter - final rough accuracy = 90%
    
    # calculate the acceleration (as the first-derivative of the velocity)
    data_phone[i]['x_acc'] = calculateDerivativeList(data_phone[i]['timestamp'], data_phone[i]['x_vel'])
    data_phone[i]['y_acc'] = calculateDerivativeList(data_phone[i]['timestamp'], data_phone[i]['y_vel']) 
    
    # scale x, y acceleration
#    data_phone[i][['x_acc', 'y_acc']] = data_phone[i][['x_acc', 'y_acc']] * 10000000.0
    
    # acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_phone[i]['filtered_x_acc'] = sig.savgol_filter(data_phone[i]['x_acc'], 15, 5)
    data_phone[i]['filtered_y_acc'] = sig.savgol_filter(data_phone[i]['y_acc'], 15, 5)
    
    
    
    #%% SMARTPHONE DATA PLOTTING
#    data_phone[i][[
#            'timestamp',
#            
##            'x_pos', 
##            'y_pos',
#            
#            'x_vel',
##            'y_vel',
#            
##            'filtered_x_vel',
##            'filtered_y_vel',
#            
##            'x_acc',
##            'y_acc',
#            
##            'filtered_x_acc',
##            'filtered_y_acc'
#            ]].plot(ax=ax, x='timestamp')
#    
    
    #%% SMARTWATCH DATA PLOTTING
    data_watch[i][[
            'timestamp',
    
#            'filtered_x_pos',
#            'filtered_y_pos',
#            'filtered_z_pos',
            
#            'x_pos', 
#            'y_pos', 
#            'z_pos',
    
#            'filtered_x_vel',
#            'filtered_y_vel',
#            'filtered_z_vel',
#            
            'x_vel', 
#            'y_vel', 
#            'z_vel',
    
#            'filtered_linear_x_acc', 
#            'filtered_linear_y_acc', 
#            'filtered_linear_z_acc',
            
#            'linear_x_acc', 
#            'linear_y_acc', 
#            'linear_z_acc',
#            
#            'filtered_x_acc',
#            'filtered_y_acc',
#            'filtered_z_acc',
            
            'x',
#            'y',
#            'z',
    
#            'magnitude',
#            'filtered_magnitude',
#            'filtered_magnitude_peaks',
            ]].plot(ax=ax, x='timestamp', linestyle=':')
    
path = files_phone[i].split("\\")
file_id = path[-1].split("_")[0:1]
plt.title("File ID: " + ''.join(file_id))