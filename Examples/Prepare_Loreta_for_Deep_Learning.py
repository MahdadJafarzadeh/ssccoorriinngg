# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:41:54 2020

CopyRight: Mahdad Jafarzadeh 2020

Automatic sleep scoring implemented by ssccoorriinngg package.

DEEP LEARNING ON **LORETA DATA** ... 

"""
#%% Reading EDF section
#####===================== Importiung libraries =========================#####
import mne
import numpy as np
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
#import tensorflow as tf
from scipy import signal

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)

#####==================== Defining required paths r=======================#####
main_path = "D:/Loreta_data/"
data_path = main_path + "/data/"

# Distinguishing patients from control group 

gp = loadtxt(main_path + "grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients

# Just keep controls 
for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))

# Init
raw_data_dic = {}
hyp_dic      = {}

#####==================== Read and analyze EDF files ====================######
Tic = time.time()
for idx, c_subj in enumerate(subj_c):
    
    print (f'Analyzing Subject Number: {c_subj}')
    
    ## Read in data
    file     = data_path + "LK_" + str(int(c_subj)) + "_1.edf"
    tic      = time.time()
    raw      = mne.io.read_raw_edf(file, preload = True)
    
    print('Time to read EDF: {}'.format(time.time()-tic))
    
    # plot signal
    #raw.plot(duration = 30)
    
    # Retrieve info
    DataInfo          = raw.info
    AvailableChannels = DataInfo['ch_names']
    fs                = int(DataInfo['sfreq'])
    
    # Filtering
    f_min = .3 #Hz
    f_max = 30 #Hz
    tic   = time.time()
    raw.filter(l_freq=f_min, h_freq=f_max)
    print('Filtering time: {}'.format(time.time()-tic))
    
    # Get filtered data
    filtered_data = raw.get_data()
    
    ## Choosing channels of interest (Fp1-M2, Fp2-M1)
    References       = ['Fp2'] # M2 and M1, respectively
    RequiredChannels = ['Fp1' ] 
    
    # Initializing index lists
    Idx = []
    Idx_Mastoids = []
    
    # Find index of required channels     
    for indx, c in enumerate(RequiredChannels):
        if c in AvailableChannels:
            Idx.append(AvailableChannels.index(c))
            
    # Find index of refernces (e.g. Mastoids) 
    for indx, c in enumerate(References):
        if c in AvailableChannels:
            Idx_Mastoids.append(AvailableChannels.index(c))

    ## Defining epoch length
    T = 30 #secs
    len_epoch   = fs * T
    start_epoch = 0
    n_channels  =  len(AvailableChannels)
       
    ## Cut tail; use modulo to find full epochs
    filtered_data = filtered_data[:, 0:filtered_data.shape[1] - filtered_data.shape[1]%len_epoch]
    
    ## Reshape data [n_channel, len_epoch, n_epochs]
    data_epoched = np.reshape(filtered_data,
                              (n_channels, len_epoch,
                               int(filtered_data.shape[1]/len_epoch)), order='F' )
    
    # Select channels of interest
    data_epoched_selected = data_epoched[Idx,:,:] - data_epoched[Idx_Mastoids,:,:]
    
    ## Read Hypnogram
    hyp = loadtxt(main_path + "hypnograms/LK_" +
                 str(int(c_subj)) + ".txt", delimiter="\t")
    
    #####================= Find order of the selected channels ===============#####  
 
    #Init
    picked_channels = []
    picked_refs     = []
    List_Channels   = []
    
    # Find main channels
    for jj,kk in enumerate(Idx):
        picked_channels = np.append(picked_channels, AvailableChannels[kk])
    
    # Show subject ID 
    print(f'subject LK {c_subj} ... Picked channels:')
    
    # Find references
    for jj,kk in enumerate(Idx_Mastoids):
        picked_refs     = np.append(picked_refs, AvailableChannels[kk])
        print(f'{str(picked_channels[jj])}-{str(picked_refs[jj])}')
    
    # Create lis of channels
    for kk in np.arange(0, len(Idx)):
        List_Channels = np.append(List_Channels, picked_channels[kk] + '-' + picked_refs[kk])
    
    #####=============== remove channels without scroing =================#####   
    
    # assign the init data and labels
    x_tmp_init = data_epoched_selected
    y_tmp_init = hyp
    
    # Ensure equalituy of length for arrays:
    Object.Ensure_data_label_length(x_tmp_init, y_tmp_init)
    
    # Remove disconnections
    x_tmp, y_tmp =  Object.remove_bad_signals(hypno_labels= y_tmp_init, 
                                                input_feats=x_tmp_init) 
    
    #####=========== Create a one hot encoding form of labels ============##### 

    # Create binary labels array
    yy = Object.One_hot_encoding(y_tmp)     
    
    # Ensure all the input labels have a class
    Object.Unlabaled_rows_detector(yy)
    
    # Defining dictionary to save hypnogram PER SUBJECT
    hyp_dic["hyp{}".format(c_subj)] = yy
     
    # Defining dictionary to save EEG raw data PER SUBJECT
    raw_data_dic["subject{}".format(c_subj)] = x_tmp
    
    # Removing variables for next iteration
    del x_tmp, x_tmp_init, y_tmp, y_tmp_init, yy, raw

# report total time for converting EDFs into arrays
print(f'Total data were read in {time.time()- Tic}')
    
#######====================== Save raw data as pickle ======================#####      
path     = main_path  
filename = "Loreta_EpochedDataForDeepLearning_Fp1-Fp2_FilteredData_091020"
Object.save_dictionary(path, filename, hyp_dic, raw_data_dic)