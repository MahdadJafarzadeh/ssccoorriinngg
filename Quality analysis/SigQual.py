# -*- coding: utf-8 -*-
"""
Copyright (C) 2020, Mahdad Jafarzadeh

THIS IS A CLASS FOR ANALYZING THE QUALITY OF SIGNAL.

Using this class, one can compare the quality of signals simultaneously 
recorded different devices.

"""
# =============================== import libs =============================== #

import mne
import numpy as np
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
import matplotlib.pyplot as plt
from scipy import signal

# =============================== defining calss ============================ #
class SigQual:
    
    # ~~~~~~~~~~~~~~~~~~~~~ Data and info reading section ~~~~~~~~~~~~~~~~~~~ #
    #%% Read text
    def read_txt(self,main_path, file_name, dtype = 'str',delimiter='\n'):
        from numpy import loadtxt
        output_file = loadtxt(main_path + file_name + ".txt", dtype = dtype, delimiter = delimiter)
        
        return output_file
    #%% Read Excel file
    def read_excel(self, main_path, filename, fileformat = '.xlsx'):
        import pandas as pd
        Output = pd.read_excel(main_path + filename + fileformat)

        return Output
    #%% Read EDF
    def read_edf_file(self, path_folder, filename, preload = True):
        
        if (path_folder + filename)[-4:]=='.edf':
            raw_data = mne.io.read_raw_edf(path_folder + filename         , preload = preload)
            
        else:
            raw_data = mne.io.read_raw_edf(path_folder + filename + ".edf", preload = preload)
        
        return raw_data
    
    #%% Extract EDF info
    def edf_info(self, data):
        
        # Extract all info
        Info = data.info
        
        # extract fs
        fs = int(Info['sfreq'])
        
        # extract available channels
        availableChannels = Info['ch_names']
        
        return Info, fs, availableChannels
    
    # ~~~~~~~~~~~~~~~~~~~~ Filter and pre-process section ~~~~~~~~~~~~~~~~~~~ #
    #%% Notch-filter    
    def NotchFilter(self, data, Fs, f0, Q):
        w0 = f0/(Fs/2)
        b, a = iirnotch(w0, Q)
        y = filtfilt(b, a, data)
        return y
    
    #%% Low-pass butterworth
    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y    
    
    #%% Band-pass Filtering section
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 2):
        nyq = 0.5 * fs
        low = lowcut /nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='band')
        #print(b,a)
        y = filtfilt(b, a, data)
        return y
    
    #%% high-pass Filtering section
    def butter_highpass_filter(self, data, highcut, fs, order):
        nyq = 0.5 * fs
        high = highcut/nyq
        b, a = butter(order, high, btype='highpass')
        y = filtfilt(b, a, data)
        return y
    #%% mne object filter
    
    def mne_obj_filter(self, data, sfreq, l_freq, h_freq, picks):
        
        filtered_sig = mne.filter.filter_data(data, sfreq, l_freq, h_freq, picks)
        
        return filtered_sig
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Resampling section ~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #%% Resampling higher freq to lower
    def resample_data(self, data1, data2, fs1, fs2):
        if fs1 != fs2:
            
            if fs1 < fs2:
                data2 = data2.resample(int(fs1), npad="auto")
                
            else:
                data1 = data1.resample(int(fs2), npad="auto")
                
        # Define resampled fs
        fs_res = np.min([fs1, fs2])
        
        return fs_res, data1, data2
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot section ~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    #%% plot EDF
    def plot_edf(self, data, higpass = .1, lowpass = 30, duration = 30, n_channels =1):
        
        data.plot(duration = duration, highpass = higpass , lowpass = lowpass, n_channels = n_channels)
        
    #%% Save plot
    def save_figure(self, directory, saving_name, dpi, saving_format = '.png',
                full_screen = False):
        if full_screen == True:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
        plt.savefig(directory+saving_name+saving_format,dpi = dpi)   
    #%% plot spectrogram
    def spectrogram_creation(self, sig1, sig2, fs, save_name, save_fig = False, dpi = 1000,\
                             save_dir = "F:\Zmax_Data\Results\SignalQualityAnalysis"):
        
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
        
        #Save figure
        if save_fig == True:
            self.save_figure(directory=save_dir, saving_name= save_name,
                         dpi=dpi, saving_format = '.png',
                         full_screen = False)
            
        #=== Maximize ====
        return f_sig1, f_sig2, Sxx_sig1, Sxx_sig2
    
    #%% Synchronization section
    def sync_data(self, fs_res, LRLR_start_zmax, LRLR_end_zmax, LRLR_start_somno, LRLR_end_somno,\
                  data_R_resampled_filtered, data_L_resampled_filtered, \
                  EEG_somno_resampled_filtered, AvailableChannels, save_name, \
                  RequiredChannels = ['F4:A1'], save_fig = False, dpi = 1000,\
                  save_dir = "F:\Zmax_Data\Results\SignalQualityAnalysis",
                  report_pearson_corr_during_sync  = False,\
                  report_spearman_corr_during_sync = False,\
                  plot_cross_corr_lag = True):
        
        """ Please note: the event detection should be preliminary given to algorithm 
        by visual inspection.
        
        events can be blinks, eye movements, etc.
        """
        # ===================== start of LRLR for sync ========================= #
    
        # Zmax
        LRLR_start_zmax = LRLR_start_zmax   #sec
        LRLR_end_zmax   = LRLR_end_zmax     #sec
        
        # Somno
        LRLR_start_somno = LRLR_start_somno #sec
        LRLR_end_somno   = LRLR_end_somno   #sec
        
        # Define a period around sync point ro perform alignment
        zmax_plotting_secs = [LRLR_start_zmax,LRLR_end_zmax]
        somno_plotting_secs = [LRLR_start_somno, LRLR_end_somno]
        
        # Finding corresponding samples of sync period
        zmax_plotting_samples  = np.arange(zmax_plotting_secs[0] *fs_res, zmax_plotting_secs[1] * fs_res)
        somno_plotting_samples = np.arange(somno_plotting_secs[0] *fs_res, somno_plotting_secs[1] * fs_res)
        
        # Convert (probable) floats into int
        somno_plotting_samples = somno_plotting_samples.astype(np.int32)
        zmax_plotting_samples  = zmax_plotting_samples.astype(np.int32)
        
        # R EEG (Zmax) --> sync period
        zmax_data_R = np.ravel(data_R_resampled_filtered)
        
        # L EEG (Zmax) --> sync period
        zmax_data_L = np.ravel(data_L_resampled_filtered)
        
        # Define channel of interest
        RequiredChannels  = RequiredChannels # main electrodes
        
        # init index of reeuired channel(s)   
        Idx               = []
        
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
        sig_somno    = Somno_reqChannel[somno_plotting_samples]
        
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
        if save_fig == True:
            self.save_figure(directory=save_dir, saving_name= save_name,
                         dpi=dpi, saving_format = '.png',
                         full_screen = False)
            
        # Report Pearson correlations during sync period
        if report_pearson_corr_during_sync == True:
            sig1 = Somno_reqChannel[somno_plotting_samples-lag]
            sig2 = sig_zmax
            self.pearson_corr(sig1, sig2)
        
        # Report spearman correlations during sync period
        if report_spearman_corr_during_sync == True:
            sig1 = Somno_reqChannel[somno_plotting_samples-lag]
            sig2 = sig_zmax
            self.spearman_corr(sig1, sig2)
        
        # Plot the cross-corr by which the lag was found
        if plot_cross_corr_lag == True:
            
            fig, ax = plt.subplots(1,1, figsize=(26, 14))
    
            ax.plot(np.arange(-len(zmax_data_L[zmax_plotting_samples])+1,len(zmax_data_L[zmax_plotting_samples])), corr, color = 'blue')
            plt.title('Cross-correlation to find lag between Zmax & Somno during eye movements', size=15)
            
            # Marking max correlation value to find lag
            ymax = np.max(np.abs(corr)) 
            
            # If negative peak, put the arrow below it
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
            
        return lag, corr, Somno_reqChannel, zmax_data_R
            
    #%% Pearson correlation
            
    def pearson_corr(self, sig1, sig2, abs_value = True, print_results = True):
        
        from scipy.stats import pearsonr

        pearson_corr,pval = pearsonr(sig1, sig2)
        
        # calculate absolute corr if needed:
        if abs_value == True:
            pearson_corr = np.abs(pearson_corr)
        
        if print_results == True:
            print(f'Pearson corr during sync period between signal1 and signal2\
              is {pearson_corr}, p-value: {pval}')
        
        return pearson_corr,pval
    
    #%% Spearman correlation
    def spearman_corr(self, sig1, sig2, abs_value = True, print_results = True):
        
        from scipy.stats import spearmanr

        spearman_corr,pval = spearmanr(sig1, sig2)
        
        # calculate absolute corr if needed:
        if abs_value == True:
            spearman_corr = np.abs(spearman_corr)
            
        if print_results == True:
            print(f'Spearman corr during sync period between signal1 and signal2\
              is {spearman_corr}, p-value: {pval}')
        
        return spearman_corr,pval
     
    #%% Plotting COMPLETE signals after synchronization
    
    def plot_full_sig_after_sync(self, LRLR_start_somno, LRLR_start_zmax, fs_res,
                                 lag, full_sig_somno_before_sync,
                                 full_sig_zmax_before_sync):
        
        # rough lag 
        rough_lag = (LRLR_start_somno - LRLR_start_zmax) * fs_res
        
        # Total lag = rough lag +- lag during sync
        total_lag = int(rough_lag - lag)
        
        # truncate the lag period from somno BEGINNING
        truncated_beginning_somno = full_sig_somno_before_sync[total_lag:]
        
        # Truncate the end of LONGER signal
        len_s = len(truncated_beginning_somno)
        len_z = len(full_sig_zmax_before_sync)
        
        # if somno data is larger
        if len_s > len_z:
            somno_final = truncated_beginning_somno[:len_z]
            zmax_final  = full_sig_zmax_before_sync
        else: 
            zmax_final  = full_sig_zmax_before_sync[:len_s]
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
    def plot_psd(self, zmax_full_sig, somno_full_sig, fs_res):
        plt.figure()
        
        # Global setting for axes values size
        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)
        
        # Plot power spectrums
        plt.psd(x=zmax_full_sig,Fs = fs_res, label = 'Zmax', NFFT = 2 ** 11, scale_by_freq= True, linewidth = 2, color = 'blue')           
        plt.psd(x=somno_full_sig,Fs = fs_res, label = 'Zmax',NFFT = 2 ** 11, scale_by_freq= True, linewidth = 2, color = 'red')     
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

    #%% save_dic
    def save_dictionary(self, path, fname, labels_dic, features_dic):
        import pickle        
        with open(path+fname+'.pickle',"wb") as f:
            pickle.dump([features_dic, labels_dic], f)