# -*- codiEEG_dsddsdsng: utf-8 -*-
"""
Created on Mon Jun 29 20:08:11 2020

@author: mahjaf
"""

#%% Import libs
#####===================== Importiung libraries =========================#####
import mne
import numpy as np
from scipy.integrate import simps
from   numpy import loadtxt
import h5py
import time
import os 
from ssccoorriinngg import ssccoorriinngg
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import tensorflow as tf
from scipy import signal
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
from scipy.stats import pearsonr
import matplotlib.mlab as mlab

#%% Read in data (Somno + Zmax)
#####=========================== Reading data ============================#####
# Main path
folder_zmax     = "F:/Zmax_Data/Zmax_Data/P_18/night2/"
folder_somno    = "F:/Zmax_Data/Somnoscreen_Data/P_18/P18 night2_B.12.12.2018/"
 
# Reading EEG left and right (Zmax)
data_L     = mne.io.read_raw_edf(folder_zmax + "EEG L.edf", preload = True)
data_R     = mne.io.read_raw_edf(folder_zmax + "EEG R.edf", preload = True)

# Read somno data    
SOMNO_path = folder_somno+"P18_night2_B_Markers_(1).edf"
EEG_somno = mne.io.read_raw_edf(SOMNO_path, preload = True)

# Reading info header (Somno)
SomnoInfo = EEG_somno.info
AvailableChannels = SomnoInfo['ch_names']
ZmaxInfo = data_R.info

# Fs
fs_zmax  = int(ZmaxInfo['sfreq'])
fs_somno = int(SomnoInfo['sfreq'])

#%% Plot filtered signals 
#####======================== Data representation ========================#####
# =============================================================================
# data_L.plot(duration = 30, highpass = .1 , lowpass = 30 )
# data_R.plot(duration = 30, highpass = .1 , lowpass = 30 )
# EEG_somno.plot(duration = 30, highpass = .1 , lowpass = 30,n_channels = 4 )
# =============================================================================

#%% Resampling higher freq to lower
if fs_zmax != fs_somno:
    
    if fs_zmax < fs_somno:
        EEG_somno = EEG_somno.resample(int(fs_zmax), npad="auto")
        
    else:
        data_L    = data_L.resample(int(fs_somno), npad="auto")
        data_R    = data_R.resample(int(fs_somno), npad="auto")
        
# Define resampled fs
fs_res = np.min([fs_zmax, fs_somno])


#%% Get data (resampled)
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = filtfilt(b, a, data)
    return y

data_L_get = data_L.get_data()
data_R_get = data_R.get_data()
data_somno_get = EEG_somno.get_data()

#%% Filtering resampled data
data_L_resampled_filtered = butter_bandpass_filter(data_L_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)
data_R_resampled_filtered = butter_bandpass_filter(data_R_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)
EEG_somno_resampled_filtered = butter_bandpass_filter(data_somno_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)

#%% Synchronization section

# ===================== start of LRLR for sync ============================== #

# Zmax
LRLR_start_zmax = 71 #sec
LRLR_end_zmax   = 91 #sec

# Somno
LRLR_start_somno = 7925 #sec
LRLR_end_somno   = 7945 #sec

# Define a period around sync point ro perform alignment
zmax_plotting_secs = [LRLR_start_zmax,LRLR_end_zmax]
somno_plotting_secs = [LRLR_start_somno, LRLR_end_somno]

# Finding corresponding samples of sync period
zmax_plotting_samples  = np.arange(zmax_plotting_secs[0] *fs_res, zmax_plotting_secs[1] * fs_res)
somno_plotting_samples = np.arange(somno_plotting_secs[0] *fs_res, somno_plotting_secs[1] * fs_res)

# Convert (probable) floats int o int
somno_plotting_samples = somno_plotting_samples.astype(np.int32)
zmax_plotting_samples  = zmax_plotting_samples.astype(np.int32)

# R EEG (Zmax) --> sync period
zmax_data_R = np.ravel(data_R_resampled_filtered)

# L EEG (Zmax) --> sync period
zmax_data_L = np.ravel(data_L_resampled_filtered)

# Define channel of interest
RequiredChannels  = ['F4:A1'] # main electrodes

# init index of reeuired channel(s)   
Idx               = []
Idx_Mastoids      = []

# Find index of required channel(s)
for indx, c in enumerate(AvailableChannels):
    if c in RequiredChannels:
        Idx.append(indx)
        
# pick Somno channel
Somno_reqChannel = EEG_somno_resampled_filtered[Idx,:]

# np.ravel somno signal(s)
Somno_reqChannel = np.ravel(Somno_reqChannel)

# plt R EEG (zmax) and required channel of Somno BEFORE sync
plt.figure()
sig_zmax     = zmax_data_R[zmax_plotting_samples]
sig_somno = Somno_reqChannel[somno_plotting_samples]

# Compute correlation
corr = signal.correlate(sig_zmax, sig_somno)

# find lag
lag = np.argmax(np.abs(corr)) - len(zmax_data_L[zmax_plotting_samples]) + 1

# Plot before lag correction
plt.plot(np.arange(0, len(zmax_plotting_samples)), sig_zmax,label = 'Zmax R EEG', color = 'black')
plt.plot(np.arange(0, len(somno_plotting_samples)), sig_somno, label = 'Somno F4', color = 'gray', linestyle = ':')
plt.title('Syncing Somno and Zmax data (Sync period only)', size = 15)

# Plot after lag correction
plt.plot(np.arange(0+lag, len(somno_plotting_samples)+lag), sig_somno, label = 'Somno F4 - synced',color = 'red')
#plt.plot(np.arange(0-lag, len(zmax_plotting_samples)-lag), sig_zmax, label = 'zmax - synced',color = 'cyan')

plt.legend(prop={"size":20})

#%% Plot cross-correlation
fig, ax = plt.subplots(1,1, figsize=(26, 14))

# Plot original Zmax sig (sync period)
ax.plot(np.arange(-len(zmax_data_R[zmax_plotting_samples])+1,len(zmax_data_R[zmax_plotting_samples])), corr, color = 'blue')
plt.title('Cross-correlation to find lag between Zmax & Somno during eye movements', size=15)

# Marking max correlation value to find lag
ymax = np.max(np.abs(corr)) 

# check if the argmax <0 --> arrow comes below figure
if np.max(np.abs(corr)) != np.max(corr) :
    ymax = -ymax
xpos = lag
xmax = lag

# Creating arrot to point to max
ax.annotate('max correlation', xy=(xmax, ymax), xytext=(xmax, ymax+ymax/10),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
plt.show()
# pearson correlation
sig_somno_new = Somno_reqChannel[somno_plotting_samples-lag]
pearson_corr,pval = pearsonr(sig_zmax, sig_somno_new)

#%% Plotting COMPLETE signals after synchronization

# rough lag 
rough_lag = (LRLR_start_somno - LRLR_start_zmax) * fs_res

# Total lag = rough lag +- lag during sync
total_lag = int(rough_lag - lag)

# truncate the lag period from somno BEGINNING
truncated_beginning_somno = Somno_reqChannel[total_lag:]

# Truncate the end of LONGER signal
len_s = len(truncated_beginning_somno)
len_z = len(zmax_data_R)

# if somno data is larger
if len_s > len_z:
    somno_final = truncated_beginning_somno[:len_z]
    zmax_final  = zmax_data_R
else: 
    zmax_final  = zmax_data_R[:len_s]
    somno_final = truncated_beginning_somno

# Calculate final length
common_length = np.min([len_s, len_z])  

# Plot truncated sigs
plt.figure()
plt.plot(np.arange(0, common_length) / fs_res / 60, zmax_final, color = 'blue', label = 'Zmax R EEG')
plt.plot(np.arange(0, common_length) / fs_res / 60, somno_final, \
         color = 'red', label = 'Somno F4-A1')
plt.title('Complete Zmax and Somno data after full sync', size = 20)
plt.xlabel('Time (mins)', size = 15)
plt.ylabel('Amplitude (v)', size = 15)
plt.legend(prop={"size":20}, loc = "upper right")

#%% Plot PSD

def spectrogram_creation(sig1,sig2, fs):
    from lspopt import spectrogram_lspopt
    import numpy as np
    import matplotlib.pyplot as plt

    #==== plot 1st sig =======   
    f, t, Sxx = spectrogram_lspopt(x=sig1, fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                   scaling='density')
    Sxx = 10 * np.log10(Sxx) #power to db
        
    # Limit Sxx to the largest freq of interest:
    f_sig1 = f[0:750]
    Sxx_sig1 = Sxx[0:750, :]
    fig, axs = plt.subplots(2,1, figsize=(26, 14))
    plt.axes(axs[0])
    
    plt.pcolormesh(t, f_sig1, Sxx_sig1)
    plt.ylabel('Frequency [Hz]', size=15)
    #plt.xlabel('Time [sec]', size=15)
    plt.title('Somnoscreeen data (F4) - Multi-taper Spectrogram', size=20)
    plt.colorbar()
    # ==== plot 2nd sig ==== #
    plt.axes(axs[1])
    f, t, Sxx = spectrogram_lspopt(x=sig2, fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                   scaling='density')
    Sxx = 10 * np.log10(Sxx) #power to db
        
    # Limit Sxx to the largest freq of interest:
    f_sig2 = f[0:750]
    Sxx_sig2 = Sxx[0:750, :]
    plt.pcolormesh(t, f_sig2, Sxx_sig2)
    plt.ylabel('Frequency [Hz]', size=15)
    plt.xlabel('Time [sec]', size=15)
    plt.title('Zmax data (EEG right) - Multi-taper Spectrogram ', size=20)

    plt.colorbar()
    #==== 1st Way =======
    
    #=== Maximize ====
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18)
    plt.show()
    #=== Maximize ====

spectrogram_creation(somno_final, zmax_final, fs = fs_res)


# =============================================================================
# #%% PERIODOGRAM
# 
# # Defining EEG bands:
# eeg_bands = {'Delta' : (0.5, 4),
#          'Theta'     : (4  , 8),
#          'Alpha'     : (8  , 11),
#          'Beta'      : (16 , 24),
#          'Sigma'     : (12 , 15),
#          'Sigma_slow': (10 , 12)}
# # Settings of peridogram    
# Window = 'hann'
# 
# # Compute pxx (SOMNO)
# fm, pxx_somno = periodogram(x = somno_full_sig, fs = fs_res , window = Window)
# 
# # Compute pxx (Zmax)
# fm, pxx_zmax = periodogram(x = zmax_full_sig, fs = fs_res , window = Window)
# freq_resolu_per= fm[1] - fm[0]
# 
# # Finding the index of different freq bands with respect to "fm" PERIODOGRAM #
# freq_ix  = dict()
# for band in eeg_bands:
#     freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
#                                (fm <= eeg_bands[band][1]))[0]    
# 
# # Periodogram
# plt.figure()
# plt.plot(fm[freq_ix['Delta']], pxx_zmax[freq_ix['Delta']], label = 'Zmax')           
# plt.plot(fm[freq_ix['Delta']], pxx_somno[freq_ix['Delta']],label = 'Somno')   
# =============================================================================

#%% Plot PSD
plt.figure()

# Global setting for axes values size
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

# Plot power spectrums
psd_z, f_psd_z = plt.psd(x=zmax_final,Fs = fs_res, label = 'Zmax', NFFT = 2 ** 11, scale_by_freq= True, linewidth = 2, color = 'blue')           
psd_s, f_psd_s = plt.psd(x=somno_final,Fs = fs_res, label = 'Zmax',NFFT = 2 ** 11, scale_by_freq= True, linewidth = 2, color = 'red')     
# ================== plot dashed lines of freq bins ========================= #

#Delta
plt.axvline(.5, linestyle = '--', color = 'black')
plt.axvline(4, linestyle = '--', color = 'black')

#Theta
plt.axvline(8, linestyle = '--', color = 'black')

# Alpha
plt.axvline(12, linestyle = '--', color = 'black')

# Title and labels
plt.title('Power spectral density throughout the night', size = 20)
plt.xlabel('Frequency (Hz)', size = 20)
plt.ylabel('Power spectral density (dB/ Hz)', size = 20)

# Legend 
plt.legend(['Zmax EEG R', 'Somno F4'], prop = {'size':20})

# Deactivate grid
plt.grid(False)

# Adding labels
plt.text(1.5, -89, 'Delta',size =18)
plt.text(5, -89, 'Theta',size =18)
plt.text(9, -89, 'Alpha',size =18)
plt.text(13, -89, 'Beta',size =18)

# Limiting x-axis to 0-30 Hz
plt.xlim([0, 30])


