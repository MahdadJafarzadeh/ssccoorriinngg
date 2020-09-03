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
#from ssccoorriinngg import ssccoorriinngg
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import tensorflow as tf
from scipy import signal
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
from scipy.stats import pearsonr, spearmanr
import matplotlib.mlab as mlab
import pandas as pd

#%% Read in data (Somno + Zmax)

#####=========================== Reading data ============================#####

# Main path
main_path   = "F:/Zmax_Data/features/"

# Read location of Somno data
subj_ids_somno  = loadtxt(main_path + "SigQual_Somno_data_loc.txt", dtype = 'str',delimiter='\n')

# Read Zmax data
subj_ids_zmax   = loadtxt(main_path + "SigQual_Zmax_data_loc.txt", dtype = 'str',delimiter='\n')

# Read subject_night id
subj_night   = loadtxt(main_path + "Subject_Night.txt", dtype = 'str',delimiter='\n')

# read event markers path to sync data
sync_markers_path= "F:/Zmax_Data/features/Sync_periods.xlsx"
event_markers = pd.read_excel(sync_markers_path)

# Define filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = filtfilt(b, a, data)
    return y

# Define spectrogram creator
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
    return f_sig1, f_sig2, Sxx_sig1, Sxx_sig2
# save figure
def save_figure(directory, saving_name, dpi, saving_format = '.png',
                full_screen = False):
    if full_screen == True:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    plt.savefig(directory+saving_name+saving_format,dpi = dpi)   

def save_dictionary(path, fname, labels_dic, features_dic):
    import pickle        
    with open(path+fname+'.pickle',"wb") as f:
        pickle.dump([features_dic, labels_dic], f)
        
#initializing dictionaries to save output
Sxx_somno_dic = dict()
Sxx_zmax_dic = dict()
f_spect_somno_dic = dict()
f_spect_zmax_dic = dict()
psd_somno_dic = dict()
psd_zmax_dic = dict()
f_psd_somno_dic = dict()
f_psd_zmax_dic = dict()

#####======================== Iterating through subjs=====================#####
subj_ids_somno = ["F:/Zmax_Data/Somnoscreen_Data/P_12/P12 night2_B.25.11.2018/P12_night2_B_markers_(1).edf"]
# Create for loop to iterate through all subjects
for idx, c_subj in enumerate(subj_ids_somno):

    # define the current zmax data
    curr_zmax  = subj_ids_zmax[idx]
    
    # define current somno data
    curr_somno = c_subj
    
    # Reading EEG left and right (Zmax)
        
    data_L     = mne.io.read_raw_edf(curr_zmax + "EEG L.edf", preload = True)
    data_R     = mne.io.read_raw_edf(curr_zmax + "EEG R.edf", preload = True)
    
    # Read somno data    
    
    EEG_somno  = mne.io.read_raw_edf(curr_somno, preload = True)
    
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
    data_L_get = data_L.get_data()
    data_R_get = data_R.get_data()
    data_somno_get = EEG_somno.get_data()
    
    #%% Filtering resampled data
    data_L_resampled_filtered = butter_bandpass_filter(data_L_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)
    data_R_resampled_filtered = butter_bandpass_filter(data_R_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)
    EEG_somno_resampled_filtered = butter_bandpass_filter(data_somno_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)

    #%% Synchronization section
    
    # ===================== start of LRLR for sync ========================= #
    
    # Zmax
    LRLR_start_zmax = event_markers['LRLR_start_zmax'][idx] #sec
    LRLR_end_zmax   = event_markers['LRLR_end_zmax'][idx] #sec
    
    # Somno
    LRLR_start_somno = event_markers['LRLR_start_somno'][idx] #sec
    LRLR_end_somno   = event_markers['LRLR_end_somno'][idx] #sec
    
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
    figure = plt.gcf()  # get current figure
    plt.xlabel('Samples',size = 15)
    plt.ylabel('Amp',size = 15)
    figure.set_size_inches(32, 18)
    
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
    #plt.plot(np.arange(0+lag, len(somno_plotting_samples)+lag), sig_somno, label = 'Somno F4 - synced',color = 'red')
    plt.plot(np.arange(0, len(somno_plotting_samples)), Somno_reqChannel[somno_plotting_samples-lag], label = 'Somno F4 - synced',color = 'red')
    #plt.plot(np.arange(0-lag, len(zmax_plotting_samples)-lag), sig_zmax, label = 'zmax - synced',color = 'cyan')
    
    plt.legend(prop={"size":20})
    
    # Save figure
    save_figure(saving_format = '.png',
                   directory="F:/Zmax_Data/Results/SignalQualityAnalysis/sync_period/",
                   saving_name = subj_night[idx], dpi = 900,
                   full_screen = False)
    
    # close current fig
    plt.close()
    
    #%% Compute correlation during sync period only
    
    sync_period_s  = Somno_reqChannel[somno_plotting_samples-lag]
    sync_period_z  = sig_zmax
    
    # compute pearson correlation 
    pearson_corr,pval_pe = pearsonr(sync_period_s, sync_period_z)
    print(f'Pearson corr during sync period between Zmax EEG R and Somno F4:A1\
          is {pearson_corr}, p-value: {pval_pe}')
          
    # Spearman Corr
    Spearman_corr,pval_sp = spearmanr(sync_period_s, sync_period_z)
    print(f'Spearman corr during sync period between Zmax EEG R and Somno F4:A1\
          is {Spearman_corr}, p-value: {pval_sp}')
    #%% Plot cross-correlation
    fig, ax = plt.subplots(1,1, figsize=(26, 14))
    
    
    ax.plot(np.arange(-len(zmax_data_L[zmax_plotting_samples])+1,len(zmax_data_L[zmax_plotting_samples])), corr, color = 'blue')
    plt.title('Cross-correlation to find lag between Zmax & Somno during eye movements', size=15)
    
    # Marking max correlation value to find lag
    ymax = np.max(np.abs(corr)) 
    if np.max(np.abs(corr)) != np.max(corr) :
        ymax = -ymax
    xpos = lag
    xmax = lag
    
    # Creating arrot to point to max
    ax.annotate('max correlation', xy=(xmax, ymax), xytext=(xmax, ymax+ymax/10),
                arrowprops=dict(facecolor='red', shrink=0.05),
                )
    
    # title, etc
    plt.title('Cross-correlation during event emergence', size = 20)
    plt.xlabel('Lag (samples)', size = 15)
    plt.ylabel('Amplitude', size = 15)
    plt.show()
    
    # Save figure
    save_figure(saving_format = '.png',
                   directory="F:/Zmax_Data/Results/SignalQualityAnalysis/Cross-corr/",
                   saving_name = subj_night[idx], dpi = 900,
                   full_screen = False)
    
    # close current fig
    plt.close()

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
    
    # compute pearson correlation 
    pearson_corr,pval_pe = pearsonr(somno_final, zmax_final)
    print(f'Pearson corr during sync period between Zmax EEG R and Somno F4:A1\
          is {pearson_corr}, p-value: {pval_pe}')
          
    # close current fig
    #plt.close()
          
    
    
    #%% Computing Coherence of signals
    plt.figure()
    coh, f = plt.cohere(somno_final, zmax_final, Fs = fs_res, NFFT = 256)
    plt.xlim([0, 30])

    #%% Plot spectrgoram of somno vs Zmax
    f_spect_s, f_spect_z, Sxx_s, Sxx_z = spectrogram_creation(somno_final, zmax_final, fs = fs_res)
    
    # Save figure
    save_figure(saving_format = '.png',
                   directory="F:/Zmax_Data/Results/SignalQualityAnalysis/spectrogram/",
                   saving_name = subj_night[idx], dpi = 900,
                   full_screen = False)
    
    # close current fig
    plt.close()
    #%% Plot PSD
    plt.figure()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(26, 14)
    
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
    
    # Save figure
    save_figure(saving_format = '.png',
                   directory="F:/Zmax_Data/Results/SignalQualityAnalysis/PSD/",
                   saving_name = subj_night[idx], dpi = 900,
                   full_screen = False)
    
    # close current fig
    plt.close()
    
    #%% Keep the PSD and spectrogram values for final normalization over subjs
    # === 1. Spectrogram:
    Sxx_somno_dic[subj_night[idx]]     = Sxx_s
    Sxx_zmax_dic[subj_night[idx]]      = Sxx_z 
    f_spect_somno_dic[subj_night[idx]] = f_spect_s
    f_spect_zmax_dic[subj_night[idx]]  = f_spect_z
    
    # === 2. PSD:
    psd_somno_dic[subj_night[idx]]     = psd_s
    psd_zmax_dic[subj_night[idx]]      = psd_z
    f_psd_somno_dic[subj_night[idx]]   = f_psd_s
    f_psd_zmax_dic[subj_night[idx]]    = f_psd_z
    
#%% Save final PSD and Freqs
path = "F:/Zmax_Data/features/"
save_dictionary(path, "SpectrogramValsSomno", Sxx_somno_dic, f_spect_somno_dic)
save_dictionary(path, "SpectrogramValsZmax", Sxx_zmax_dic, f_spect_zmax_dic)
save_dictionary(path, "PSDValsZmax", psd_zmax_dic, f_psd_zmax_dic)
save_dictionary(path, "PSDValsSomno", psd_somno_dic, f_psd_somno_dic)

