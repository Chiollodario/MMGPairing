import pandas as pd
import numpy as np
import scipy.signal as sig
from scipy.fftpack import fftfreq, fft, ifft
from scipy.integrate import romb
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import smoothed_zscore as sz
import glob

#plt.close('all')

# DataFrame collection from files
files_phone = glob.glob('C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-09-21_7_smartphone_sample.csv')
files_watch = glob.glob('C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-09-21_7_watch_sample.csv')

data_phone = [pd.read_csv(x) for x in files_phone] # List comprehension
data_watch = [pd.read_csv(x) for x in files_watch]

# global variable for the cutoff frequency used in the low-pass filter
cutoff_freq = 50

# global viarables for tweaking the s_zscore thresholding_algo (lag, threshold, and influence parameters)
# and for plotting in a more understandable way the peaks (peak_multiplicator)
# N.B: does NOT apply to the first use of the algorithm (i.e: the very first peak detection)
peak_multiplicator = 100
lag = 2
threshold = 3
influence = 0.5

# global viarables for the Grey-code similarity calculation 
# used by "grey_code_similarity" method
max_greycode_window = 0
error_threshold = 0

# dictionary for the Grey-code extraction
grey_code_dict =	{
  '1': '01',
  '0': '00',
  '-1': '11'
}

# utlity function for getting necessary info for the integral calculation
def f(x, key):
    if (x is None or key is None):
        ValueError(" f:  parameters must non-null ")
    index = pd.Index(data_watch[i]['timestamp']).get_loc(x)
    value = data_watch[i][key][index]
    return value

# integral calculated as the area beneath the graph, for each datapoint couple
def calculateIntegralList(timestampList, key):
    if (timestampList is None or len(timestampList)==0 or key is None):
        ValueError(" f:  parameters must be non-null or >0 ")
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
    if (timestampList is None or len(timestampList)==0 or toCalcList is None or len(toCalcList)==0):
        ValueError(" calculateDerivativeList:  invalid parameters ")
    toCalc = np.asarray(toCalcList)
    resultList = []    
    resultList.append(0)
    for i in range(len(toCalc)):
        if i == len(toCalc)-1: 
            break
        resultList.append((toCalc[i+1] - toCalc[i])/(timestampList[i+1] - timestampList[i]))
    return resultList

# function for first peak detection of a signal
def findFirstPeakBeginning(a):
    if (a is None or len(a)==0):
        ValueError(" findFirstPeakBeginning:  invalid parameters ")
    i = 0
    while(i<len(a) and a[i]!=1):
        i += 1
    return i

# detect the last datapoint of the first peak
def findFirstPeakEnd(a, j):
    if (a is None or len(a)==0 or j is None or j<0):
        ValueError(" findFirstPeakEnd:  invalid parameters ")
    i = j
    while(i<len(a) and a[i]==1):
        i += 1
    return i-1

# detect the the max value of the first peak's datapoints
def findPeakMaxValueIndex(a):
    if (a is None or len(a)==0):
        ValueError(" findPeakMaxValueIndex:  invalid parameters ")
    return a.idxmax();

# search for the closest value in a given array 
def find_nearest(array, value):
    if (a is None or len(a)==0 or value is None):
        ValueError(" find_nearest:  invalid parameters ")
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# rudimentary low-pass filter who removes frequencies above a cutoff threshold
def apply_lowpass_filter(a, cutoff_freq):
    if (a is None or len(a)==0 or cutoff_freq is None or cutoff_freq<0):
        ValueError(" apply_lowpass_filter:  invalid parameters ")
    a[cutoff_freq:] = 0
    return a

# extract a 2-bit Grey code from each peak of the sample
def extract_grey_code(a):
    if (a is None or len(a)==0):
        ValueError(" extract_grey_code:  invalid parameter ")
    bits_str = np.array([], dtype=str)
    bits = np.array([], dtype=int)    
    for i in range(len(a)):
        res = grey_code_dict.get(str(int(a[i])))
        bits_str = np.append(bits_str, res)
    string = ''.join(bits_str)
    # needed in order not to lose the possible initial bit 0
    for i in range(len(string)):
        bits = np.append(bits, int(string[i]))
    return bits

# returns the similarity (min 0, max 1) of the Grey-codes passed as parameters.
# This a two-pass algorithm: a, b are also inverted for the check.
# error_threshold: can be used to reject/accept the authentication
# max_window: is the max sliding window allowed for searching the best similarity (and narrow down lag problems among signals)
def grey_code_similarity(a, b, error_threshold, max_window):
    if (a is None or len(a)==0 or b is None or len(b)==0 or error_threshold is None or error_threshold <0 or max_window is None or max_window <0):
        ValueError(" grey_code_similarity:  invalid parameters ")
    return max(grey_code_similarity_split(a,b,max_window), grey_code_similarity_split(b,a,max_window))

def grey_code_similarity_split(a, b, max_window):
    if (a is None or len(a)==0 or b is None or len(b)==0 or max_window is None or max_window <0):
        ValueError(" grey_code_similarity:  invalid parameters ")
    max_similarity = 0
    current_window = 0
    matching_bits = 0
    best_window = 0
    
    while(current_window<=max_window):
        i = current_window
        j = 0        
        while(i<len(a)):
            if (a[j]==b[i]):
               matching_bits+=1
            i+=1
            j+=1      
        if(matching_bits/(len(a)-current_window)>max_similarity):
            max_similarity = matching_bits/(len(a)-current_window)
            best_window = current_window        
        matching_bits = 0
        current_window+=1
    return max_similarity    



# === some commented parts are left for research purpose (e.g: in case they are needed for different tries)
# WATCH DATA ANALYSIS
if (files_watch is None or len(files_watch)<0):
    ValueError(" WATCH DATA ANALYSIS:  files_watch parameter not valid ")
for i in range(len(files_watch)):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    
    #%% WATCH - PEAK DETECTION
    # 1st step: Savitzky–Golay filter - final rough accuracy = 90%
    # 2st step: Smoothed z-score algorithm - final rough accuracy = low since it returns a digital result
    
    # calculate the linear acceleration for each axis    
    data_watch[i]['temp_linear_x_acc'] = data_watch[i]['x_acc'] - data_watch[i]['x_acc'].mean()
    data_watch[i]['temp_linear_y_acc'] = data_watch[i]['y_acc'] - data_watch[i]['y_acc'].mean()
    data_watch[i]['temp_linear_z_acc'] = data_watch[i]['z_acc'] - data_watch[i]['z_acc'].mean()
    
    # acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_x_acc'] = sig.savgol_filter(data_watch[i]['temp_linear_x_acc'], 15, 5)
    data_watch[i]['filtered_y_acc'] = sig.savgol_filter(data_watch[i]['temp_linear_y_acc'], 15, 5)
    data_watch[i]['filtered_z_acc'] = sig.savgol_filter(data_watch[i]['temp_linear_z_acc'], 15, 5)
    
    # calculate the Euclidean norm = 2-norm (that is: the magnitude)
    a = np.column_stack((data_watch[i]['filtered_x_acc'], data_watch[i]['filtered_y_acc'], data_watch[i]['filtered_z_acc']))
    b = np.zeros(a.shape[0])
    for j in range(a.shape[0]):
        b[j] = np.linalg.norm(a[j]) # Euclidean norm = 2-norm 
    data_watch[i]['magnitude'] = b
    
    # magnitude noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_magnitude'] = sig.savgol_filter(b, 15, 5)    
       
    # peak detection
    # y = data to analyze
    # lag = the lag of the moving window 
    # threshold = the z-score at which the algorithm signals and influence
    # influence = (between 0 and 1) of new signals on the mean and standard deviation
    smoothed_zscore = sz.thresholding_algo(data_watch[i]['filtered_magnitude'], 4, 9, 0) # thresholding_algo(y, lag, threshold, influence)
    data_watch[i]['filtered_magnitude_peaks'] = smoothed_zscore.get('signals')
    
    
    #%% WATCH - SIGNAL SYNCHRONISATION
    
    # first peak detection for synchronising smartwatch and smartphone signals
    firstpeakbeginning_index = findFirstPeakBeginning(np.array(data_watch[i]['filtered_magnitude_peaks'])) # find out the beginning of the very first peak
    firstpeakend_index = findFirstPeakEnd(np.array(data_watch[i]['filtered_magnitude_peaks']), firstpeakbeginning_index) # find out the end of the very first peak    
    firstpeak_index = findPeakMaxValueIndex(data_watch[i]['filtered_magnitude'][firstpeakbeginning_index:firstpeakend_index+1])
    
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
    data_watch[i]['linear_x_acc'] = data_watch[i]['x_acc'] - data_watch[i]['x_acc'][firstpeak_index:final_index+1].mean()
    data_watch[i]['linear_y_acc'] = data_watch[i]['y_acc'] - data_watch[i]['y_acc'][firstpeak_index:final_index+1].mean()
    data_watch[i]['linear_z_acc'] = data_watch[i]['z_acc'] - data_watch[i]['z_acc'][firstpeak_index:final_index+1].mean()
    
    # linear acceleration noise filtering (through built-in Savitzky-Golay filter)
    data_watch[i]['filtered_linear_x_acc'] = sig.savgol_filter(data_watch[i]['linear_x_acc'], 15, 5)
    data_watch[i]['filtered_linear_y_acc'] = sig.savgol_filter(data_watch[i]['linear_y_acc'], 15, 5)
    data_watch[i]['filtered_linear_z_acc'] = sig.savgol_filter(data_watch[i]['linear_z_acc'], 15, 5)
    
    # FFT of the linear acceleration 
    data_watch[i]['watch_linear_x_acc_fft'] = fft(np.array(data_watch[i]['linear_x_acc']))
    data_watch[i]['watch_linear_y_acc_fft'] = fft(np.array(data_watch[i]['linear_y_acc']))
    # rudimentary low-pass filter on the FFT of the linear acceleration
    data_watch[i]['watch_linear_x_acc_fft_lp'] = apply_lowpass_filter(data_watch[i]['watch_linear_x_acc_fft'],cutoff_freq)
    data_watch[i]['watch_linear_y_acc_fft_lp'] = apply_lowpass_filter(data_watch[i]['watch_linear_y_acc_fft'],cutoff_freq)
    # Inverse FFT (lp stands for low-passed)
    data_watch[i]['watch_linear_x_acc_lp'] = ifft(data_watch[i]['watch_linear_x_acc_fft_lp'])
    data_watch[i]['watch_linear_y_acc_lp'] = ifft(data_watch[i]['watch_linear_y_acc_fft_lp'])
    
    
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
    
    # comment the next three lines if don't want to FFT over filtered values
#    temp_x_vel = sig.savgol_filter(data_watch[i]['x_vel'], 15, 5)
#    temp_y_vel = sig.savgol_filter(data_watch[i]['y_vel'], 15, 5)
#    temp_z_vel = sig.savgol_filter(data_watch[i]['z_vel'], 15, 5)
    
    # FFT of the velocity
    
    # Remove all the NaNs from the array replacing them with zero
    temp_x_vel = np.nan_to_num(temp_x_vel)
    temp_y_vel = np.nan_to_num(temp_y_vel)
    data_watch[i]['watch_x_vel_fft'] = fft(np.array(temp_x_vel))
    data_watch[i]['watch_y_vel_fft'] = fft(np.array(temp_y_vel))
    # rudimentary low-pass filter on the FFT of the velocity
    data_watch[i]['watch_x_vel_fft_lp'] = apply_lowpass_filter(data_watch[i]['watch_x_vel_fft'], cutoff_freq)
    data_watch[i]['watch_y_vel_fft_lp'] = apply_lowpass_filter(data_watch[i]['watch_y_vel_fft'], cutoff_freq)
    # Inverse FFT (lp stands for low-passed)
    data_watch[i]['watch_x_vel_lp'] = ifft(data_watch[i]['watch_x_vel_fft_lp'])
    data_watch[i]['watch_y_vel_lp'] = ifft(data_watch[i]['watch_y_vel_fft_lp'])
    
    
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
if (files_phone is None or len(files_phone)<0):
    ValueError(" PHONE DATA ANALYSIS:  files_phone parameter not valid ")
for i in range(len(files_phone)):    
    
    #%% PHONE - GET POSITION FROM API
    # rough accuracy = 100%
    
    # scale x, y position 
    data_phone[i][['x_pos', 'y_pos']] = data_phone[i][['x', 'y']] #* 2000    
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
#    data_phone[i]['y_vel'] = data_phone[i]['y_vel'] * -1
    
    # scale x, y velocity
    data_phone[i][['x_vel', 'y_vel']] = data_phone[i][['x_vel', 'y_vel']] /50.0

    # realign the y axis (smartphones have a different y-axis direction)
    data_phone[i]['y_vel'] = data_phone[i]['y_vel'] * -1
    
    # velocity noise filtering (through built-in Savitzky-Golay filter)
    data_phone[i]['filtered_x_vel'] = sig.savgol_filter(data_phone[i]['x_vel'], 15, 5)
    data_phone[i]['filtered_y_vel'] = sig.savgol_filter(data_phone[i]['y_vel'], 15, 5)
    
    # FFT of the velocity
    data_phone[i]['phone_x_vel_fft'] = fft(np.array(data_phone[i]['x_vel']))
    data_phone[i]['phone_y_vel_fft'] = fft(np.array(data_phone[i]['y_vel']))
    # rudimentary low-pass filter on the FFT of the velocity
    data_phone[i]['phone_x_vel_fft_lp'] = apply_lowpass_filter(data_phone[i]['phone_x_vel_fft'],cutoff_freq)
    data_phone[i]['phone_y_vel_fft_lp'] = apply_lowpass_filter(data_phone[i]['phone_y_vel_fft'],cutoff_freq)
    # Inverse FFT (lp stands for low-passed)
    data_phone[i]['phone_x_vel_lp'] = ifft(data_phone[i]['phone_x_vel_fft_lp'])
    data_phone[i]['phone_y_vel_lp'] = ifft(data_phone[i]['phone_y_vel_fft_lp'])
    
    
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
    
    # FFT of the acceleration 
    data_phone[i]['phone_x_acc_fft'] = fft(np.array(data_phone[i]['x_acc']))
    data_phone[i]['phone_y_acc_fft'] = fft(np.array(data_phone[i]['y_acc']))
    # rudimentary low-pass filter on the FFT of the acceleration
    data_phone[i]['phone_x_acc_fft_lp'] = apply_lowpass_filter(data_phone[i]['phone_x_acc_fft'],cutoff_freq)
    data_phone[i]['phone_y_acc_fft_lp'] = apply_lowpass_filter(data_phone[i]['phone_y_acc_fft'],cutoff_freq)
    # Inverse FFT (lp stands for low-passed)
    data_phone[i]['phone_x_acc_lp'] = ifft(data_phone[i]['phone_x_acc_fft_lp'])
    data_phone[i]['phone_y_acc_lp'] = ifft(data_phone[i]['phone_y_acc_fft_lp'])
    
    
    #%% GREY-CODE EXTRACTION
    
    # in order to have the same code length it is necessary to RESAMPLE THE WATCH SIGNALS
    # N.B: accelerometer usually samples at a higher frequency than the Android Motion API
    phone_row_number = data_phone[i].shape[0]
    max_greycode_window = int(phone_row_number/10)
    watch_linear_x_acc_lp_resampled = sig.resample(data_watch[i]['watch_linear_x_acc_lp'][firstpeak_index:final_index+1], phone_row_number)
    watch_linear_y_acc_lp_resampled = sig.resample(data_watch[i]['watch_linear_y_acc_lp'][firstpeak_index:final_index+1], phone_row_number)
    watch_x_vel_lp_resampled = sig.resample(data_watch[i]['watch_x_vel_lp'][firstpeak_index:final_index+1], phone_row_number)
    watch_y_vel_lp_resampled = sig.resample(data_watch[i]['watch_y_vel_lp'][firstpeak_index:final_index+1], phone_row_number)
    
    # Grey-code extraction - WATCH
    # acceleration
    s_zscore_watch_x_acc_lp = sz.thresholding_algo(watch_linear_x_acc_lp_resampled, lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_watch_y_acc_lp = sz.thresholding_algo(watch_linear_y_acc_lp_resampled, lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_watch_x_acc_lp_peaks = s_zscore_watch_x_acc_lp.get('signals') * peak_multiplicator
    s_zscore_watch_y_acc_lp_peaks = s_zscore_watch_y_acc_lp.get('signals') * peak_multiplicator    
    watch_x_acc_greycode = extract_grey_code(s_zscore_watch_x_acc_lp_peaks / peak_multiplicator)
    watch_y_acc_greycode = extract_grey_code(s_zscore_watch_y_acc_lp_peaks / peak_multiplicator)

    # velocity
    s_zscore_watch_x_vel_lp = sz.thresholding_algo(watch_x_vel_lp_resampled, lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_watch_y_vel_lp = sz.thresholding_algo(watch_y_vel_lp_resampled, lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_watch_x_vel_lp_peaks = s_zscore_watch_x_vel_lp.get('signals') * peak_multiplicator
    s_zscore_watch_y_vel_lp_peaks = s_zscore_watch_y_vel_lp.get('signals') * peak_multiplicator
    watch_x_vel_greycode = extract_grey_code(s_zscore_watch_x_vel_lp_peaks / peak_multiplicator)
    watch_y_vel_greycode = extract_grey_code(s_zscore_watch_y_vel_lp_peaks / peak_multiplicator)
    
    # Grey-code extraction - PHONE
    # velocity
    s_zscore_phone_x_vel_lp = sz.thresholding_algo(data_phone[i]['phone_x_vel_lp'], lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_phone_y_vel_lp = sz.thresholding_algo(data_phone[i]['phone_y_vel_lp'], lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_phone_x_vel_lp_peaks = s_zscore_phone_x_vel_lp.get('signals') * peak_multiplicator
    s_zscore_phone_y_vel_lp_peaks = s_zscore_phone_y_vel_lp.get('signals') * peak_multiplicator   
    phone_x_vel_greycode = extract_grey_code(s_zscore_phone_x_vel_lp_peaks / peak_multiplicator)
    phone_y_vel_greycode = extract_grey_code(s_zscore_phone_y_vel_lp_peaks / peak_multiplicator)    
    
    # acceleration
    s_zscore_phone_x_acc_lp = sz.thresholding_algo(data_phone[i]['phone_x_acc_lp'], lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_phone_y_acc_lp = sz.thresholding_algo(data_phone[i]['phone_y_acc_lp'], lag, threshold, influence) # thresholding_algo(y, lag, threshold, influence)
    s_zscore_phone_x_acc_lp_peaks = s_zscore_phone_x_acc_lp.get('signals') * peak_multiplicator
    s_zscore_phone_y_acc_lp_peaks = s_zscore_phone_y_acc_lp.get('signals') * peak_multiplicator
    phone_x_acc_greycode = extract_grey_code(s_zscore_phone_x_acc_lp_peaks / peak_multiplicator)
    phone_y_acc_greycode = extract_grey_code(s_zscore_phone_y_acc_lp_peaks / peak_multiplicator)
    
    
    #%% SMARTPHONE DATA PLOTTING    
    
    data_phone[i][[
            'timestamp',
            
#            'x_pos', 
#            'y_pos',
                       
#            'filtered_x_vel',
#            'filtered_y_vel',
            
#            'x_vel',
#            'y_vel',
            
#            'filtered_x_acc',
            'filtered_y_acc',
            
#            'x_acc',
#            'y_acc'
            
            ]].plot(ax=ax1, color='r', x='timestamp')


    #%% SMARTWATCH DATA PLOTTING
    
    path = files_phone[i].split("\\")
    file_data = path[-1].split("_")[0]
    file_id = path[-1].split("_")[1]
    title = "File ID: " + ''.join(file_data) + "_" + ''.join(file_id)
    
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
#            'x_vel', 
#            'y_vel', 
#            'z_vel',
    
#            'filtered_linear_x_acc', 
            'filtered_linear_y_acc', 
#            'filtered_linear_z_acc',
            
#            'linear_x_acc', 
#            'linear_y_acc', 
#            'linear_z_acc',
#            
#            'filtered_x_acc',
#            'filtered_y_acc',
#            'filtered_z_acc',
            
#            'x_acc',
#            'y_acc',
#            'z_acc',
    
#            'magnitude',
#            'filtered_magnitude',
#            'filtered_magnitude_peaks'
            
            ]].plot(ax=ax1, x='timestamp',  title=title+'\n\nPlot before FFT')
    
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Amplitude')
    
    #%% SMARTWATCH-SMARTPHONE FFT/IFFT DATA PLOTTING
    
    data_phone[i][[
            'timestamp',
            
#            'phone_x_vel_fft',
#            'phone_y_vel_fft',
#            'phone_x_vel_fft_lp',
#            'phone_y_vel_fft_lp',
#            'phone_x_vel_lp',
#            'phone_y_vel_lp',
#            
#            'phone_x_acc_fft',
#            'phone_y_acc_fft',
#            'phone_x_acc_fft_lp',
#            'phone_y_acc_fft_lp',
#            'phone_x_acc_lp',
            'phone_y_acc_lp',
            
    ]].plot(ax=ax2, color='r', label='phone', x='timestamp')
    
    data_watch[i][[
            'timestamp',
            
#            'watch_x_vel_fft',
#            'watch_y_vel_fft',
#            'watch_x_vel_fft_lp',
#            'watch_y_vel_fft_lp',
#            'watch_x_vel_lp',
#            'watch_y_vel_lp',
#            
#            'watch_linear_x_acc_fft',
#            'watch_linear_y_acc_fft',
#            'watch_linear_x_acc_fft_lp',
#            'watch_linear_y_acc_fft_lp',
#            'watch_linear_x_acc_lp',
            'watch_linear_y_acc_lp',
            
    ]].plot(ax=ax2, color='b', linestyle=':', label='watch', x='timestamp')
    
    x_axis = np.arange(phone_row_number)
    ax3.plot(
            x_axis,
             
#             watch_x_vel_lp_resampled,
#             s_zscore_watch_x_vel_lp_peaks,
#             data_phone[i]['phone_x_vel_lp'],
#             s_zscore_phone_x_vel_lp_peaks,
             
#             watch_y_vel_lp_resampled,
#             s_zscore_watch_y_vel_lp_peaks,
#             data_phone[i]['phone_y_vel_lp'],
#             s_zscore_phone_x_vel_lp_peaks,
             
#             watch_linear_x_acc_lp_resampled,
             s_zscore_watch_x_acc_lp_peaks,
#             data_phone[i]['phone_x_acc_lp'],
             s_zscore_phone_x_acc_lp_peaks,
             
#             watch_linear_y_acc_lp_resampled,
#             s_zscore_watch_y_acc_lp_peaks,
#             data_phone[i]['phone_y_acc_lp'],
#             s_zscore_phone_y_acc_lp_peaks
             )
    
    plt.xlabel('Timestamp')
    plt.ylabel('Amplitude')
    plt.title('Plot after FFT')
    plt.legend()
    
    print(grey_code_similarity(
            
#            watch_x_acc_greycode,
#            watch_y_acc_greycode,
            watch_x_vel_greycode,
#            watch_y_vel_greycode,
            
#            phone_x_acc_greycode,
#            phone_y_acc_greycode,
            phone_x_vel_greycode,
#            phone_y_vel_greycode,
            
            error_threshold,
            max_greycode_window
    ))