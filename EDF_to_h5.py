# -*- coding: utf-8 -*-
"""
Preprocessing script
Created: 2020/04/18
                             
By: mahjaf

This script reads EDF files of interest and their corresponding labels (sleep stages)
Then Save them PER SUBJECT as a .h5 file to make the procesing faster.
"""

## Install MNE package from here: https://mne.tools/dev/install/mne_python.html

import mne
import numpy as np
from   numpy import loadtxt
import h5py
import time
import os 
## Read in patient labels
pat_labels = loadtxt("P:/3013080.02/ml_project/patient_labels.txt", delimiter="\t", skiprows = 1)

# Distinguishing patients from control group
gp = loadtxt("P:/3013080.02/ml_project/grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients

for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))
    elif c[1] == 'CC':
        pass
    else:
        subj_p.append(int(c[0]))

# Response to medication        
response = pat_labels[:,1]

for idx, c_subj in enumerate(subj_c):
    print (f'Analyzing Subject Number: {c_subj}')
    ## Read in data
    file     = "P:/3013080.02/ml_project/test_data/LK_" + str(int(c_subj)) + "_1.edf"
    tic      = time.time()
    data     = mne.io.read_raw_edf(file)
    #data.plot(duration=30, lowpass = 30, highpass = .1)
    raw_data = data.get_data()
    print('Time to read EDF: {}'.format(time.time()-tic))
    
    # Retrieve info
    DataInfo          = data.info
    AvailableChannels = DataInfo['ch_names']
    fs                = int(DataInfo['sfreq'])
    
    # Filtering
    #tic      = time.time()
    #raw_data = butter_bandpass_filter(raw_data,fs=fs,lowcut=.3,highcut= 20)
    #print('Filtering time: {}'.format(time.time()-tic))
    
    ## Choosing channels of interest (which ones to choose?)
    Mastoids         = ['Fp2'] # Reference electrode
    RequiredChannels = ['Fp1'] # test
    # Find index of required channels
    Idx = []
    Idx_Mastoids = []
    
    for indx, c in enumerate(AvailableChannels):
        if c in RequiredChannels:
            Idx.append(indx)
        elif c in Mastoids:
            Idx_Mastoids.append(indx)
    
    ## Sampling rate is 200hz; thus 1 epoch(30s) is 6000 samples
    T = 30 #secs
    len_epoch   = fs * T
    start_epoch = 0
    n_channels  =  len(AvailableChannels)
       
    ## Cut tail; use modulo to find full epochs
    raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
    
    ## Reshape data [n_channel, len_epoch, n_epochs]
    data_epoched = np.reshape(raw_data,
                              (n_channels, len_epoch,
                               int(raw_data.shape[1]/len_epoch)), order='F' )
    data_label   = np.ones((data_epoched.shape[2],1))*response[idx]
    
    ## Read in hypnogram data
    #1. Control:
    hyp = loadtxt("P:/3013065.04/Depressed_Loreta/hypnograms/LK_" + 
                str(int(c_subj)) + ".txt", delimiter="\t")
    #2. Patients:
    #hyp = loadtxt("P:/3013065.04/Depressed_Loreta/hypnograms/LP_" + 
    #              str(int(c_subj)) + "_1.txt", delimiter="\t")
    
    ### Create sepereate data subfiles based on hypnogram (N1, N2, N3, NREM, REM) 
    tic      = time.time()
    
    # Seprate channels of interest: (fp1, fp2) - non-referenced
    '''### IMPORTANT: Choose the first line below for non-referenced data ### 
    ### Choose the second line for referenced data to Mastoids ###'''
    
    #data_epoched_selected = data_epoched[Idx]
    data_epoched_selected = data_epoched[Idx] - data_epoched[Idx_Mastoids]
    print('Time to split sleep stages per epoch: {}'.format(time.time()-tic))
    
    ### Check existence of required directories for saving files

    tic      = time.time()
    Channels = 'Fp1-Fp2/'
    for jj in np.arange(len(RequiredChannels)):
        if not os.path.exists('D:/1D_TimeSeries/raw_EEG/full/' + Channels):
            os.makedirs('1D_TimeSeries/raw_EEG/full/'  +Channels)
                
    # Task finished: generate message            
    print('Required folders are created in {} secs'.format(time.time()-tic))
                  
    '''## SAVING FORMAT:
    #FILE NAME: LK_SubjectNo_(1:pre-medication or 2:post-medication)
    PLEASE NOTE: the file name (either LP_ or LK_) should be set manually! '''
    
    tic = time.time()  
    
    # Define saving directory
    fname = ('D:/1D_TimeSeries/raw_EEG/full/' +Channels +
            '/LK_'+ str(int(c_subj)) + '_1')
    
    with h5py.File((fname + '.h5'), 'a') as wf:
        # Save data from Fp1-Fp2
        dset = wf.create_dataset('data_fp1-fp2', data_epoched_selected.shape,
                                 data=data_epoched_selected)
        # Save corresponding hypnogram
        dset = wf.create_dataset('hypnogram', hyp.shape, data=hyp)
        
        
        
        
    print('Time to save H5: {}'.format(time.time()-tic))
