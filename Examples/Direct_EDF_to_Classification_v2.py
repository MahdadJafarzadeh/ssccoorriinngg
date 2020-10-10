# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:07:37 2020

@author: mahjaf

Using this code, one can directly feed in EDF data and select channels of interest
to perform classification. 

Please Note: we recommend to use "EDF_to_h5.py" to firstly convert EDF into
a lighter data format to accelerate computations; however, this code is only
meant for those that want to skip this conversion and directly choose EDF data
as input.
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

#####===================== Reading EDF data files=========================#####

#pat_labels = loadtxt("/project/3013080.02/ml_project/patient_labels.txt", delimiter="\t", skiprows = 1)

#####============= Distinguishing patients from control group=============#####
main_path = "D:/Loreta_data/"
data_path = main_path + "/data/"
gp = loadtxt(main_path + "grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients

for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))
    elif c[1] == 'CC':
        pass
    else:
        subj_p.append(int(c[0]))

# Initialization
subjects_dic     = {}
hyp_dic          = {}
metrics_per_fold = {}

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)


tic_tot = time.time()
#####============= Iterate through each subject to find data =============#####

for idx, c_subj in enumerate(subj_c):
    print (f'Analyzing Subject Number: {c_subj}')
    ## Read in data
    file     = data_path + "LK_" + str(int(c_subj)) + "_1.edf"
    tic      = time.time()
    data     = mne.io.read_raw_edf(file)

    raw_data = data.get_data()
    print('Time to read EDF: {}'.format(time.time()-tic))
    
#####=================Retrieving information from data====================#####
    
    DataInfo          = data.info
    AvailableChannels = DataInfo['ch_names']
    fs                = int(DataInfo['sfreq'])
    
#####==================Choosing channels of interest======================#####
    
    # 1. The channels that need to be referenced
    Mastoids          = ['Fp2'] # Reference electrodes
    RequiredChannels  = ['Fp1'] # main electrodes
    # 2. Channels that don't need to be referenced: --> Deactive
   
    Idx               = []
    Idx_Mastoids      = []
    
#####================= Find index of required channels ===================#####
    
    for indx, c in enumerate(AvailableChannels):
        if c in RequiredChannels:
            Idx.append(indx)
        elif c in Mastoids:
            Idx_Mastoids.append(indx)

#####===== Sampling rate is 200hz; thus 1 epoch(30s) is 6000 samples =====#####
            
    T = 30 #secs
    len_epoch   = fs * T
    start_epoch = 0
    n_channels  =  len(AvailableChannels)
       
#####============ Cut tail; use modulo to find full epochs ===============#####

    raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
    
#####========== Reshape data [n_channel, len_epoch, n_epochs] ============#####
    data_epoched = np.reshape(raw_data,
                              (n_channels, len_epoch,
                               int(raw_data.shape[1]/len_epoch)), order='F' )
    
#####===================== Reading hypnogram data ========================#####

    hyp = loadtxt(main_path + "hypnograms/LK_" +
                 str(int(c_subj)) + ".txt", delimiter="\t")
    
    ### Create sepereate data subfiles based on hypnogram (N1, N2, N3, NREM, REM) 
    tic      = time.time()
#####================= Concatenation of selected channels ================#####   
    # Calculate referenced channels: 
    data_epoched_selected = data_epoched[Idx] - data_epoched[Idx_Mastoids]
    # show picked channels for analysis
    picked_channels = []
    for jj,kk in enumerate(Idx):
        picked_channels = np.append(picked_channels, AvailableChannels[kk])
        
    print(f'subject LK {c_subj} --> detected channels: {picked_channels}')
    # Add non-referenced channels:  --> Deactive
    #data_epoched_selected_ = np.concatenate([data_epoched_selected, data_epoched_nonref], 0)
    print('Time to split sleep stages per epoch: {}'.format(time.time()-tic))
    
    #%% Analysis section
#####================= remove bad chanbnels and arousals =================#####   
    
    # assign the proper data and labels
    x_tmp_init = data_epoched_selected
    y_tmp_init = hyp
    # Remove bad signals (hyp == -1)
    x_tmp, y_tmp =  Object.remove_bad_signals(hypno_labels = y_tmp_init,
                                              input_feats = x_tmp_init)
    
#####============= remove stages contaminated with arousal ===============#####      
    
    x_tmp, y_tmp = Object.remove_artefact(hypno_labels = y_tmp, input_feats = x_tmp)
    # Create binary labels array
    yy = Object.binary_labels_creator_categories(y_tmp)

    # Initialize feature array:
    Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
      
#####================== Extract the relevant features ====================#####    
    
    for k in np.arange(np.shape(data_epoched_selected)[0]):
        
        feat_temp         = Object.FeatureExtraction_per_subject(Input_data = x_tmp[k,:,:])
        Feat_all_channels = np.column_stack((Feat_all_channels,feat_temp))
        
    toc = time.time()
    print(f'Features of subject {c_subj} were successfully extracted in: {toc-tic} secs')
    
    # Defining dictionary to save features PER SUBJECT
    subjects_dic["subject{}".format(c_subj)] = Feat_all_channels
    
    # Defining dictionary to save hypnogram PER SUBJECT
    hyp_dic["hyp{}".format(c_subj)] = yy
    
#####=============== Removing variables for next iteration ===============#####      
    del x_tmp, y_tmp, feat_temp, yy
    toc = time.time()
    print(f'Features and hypno of subject {c_subj} were successfully added to dictionary')
    
    print('Feature extraction of subject {c_subj} has been finished.')   

print('Total feature extraction of subjects took {tic_tot - time.time()} secs.')
#%% Save created features and labels
#####====================== Save extracted features ======================#####      

path     = '/project/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
filename = 'sleep_scoring_NoArousal_Fp1-Fp2_54feats'
Object.save_dictionary(path, filename, hyp_dic, subjects_dic)


#%% Load featureset and labels
path                  = '/project/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
filename              = 'sleep_scoring_NoArousal_Fp1-Fp2_54feats'
subjects_dic, hyp_dic = Object.load_dictionary(path, filename)

#%% Create leave-one-out cross-validation
 # Define counter of metrics per fold

c = 1

for idx, c_subj in enumerate(subj_c):
    print(f'test subject is {c_subj} ...')
    ### assigning strings of TEST set
    # 1. test hypno
    str_test_hyp  = 'hyp' + str(c_subj)
    #2. test featureset
    str_test_feat = 'subject' + str(c_subj)
    # creating test arrays
    X_test        =  subjects_dic[str_test_feat]
    y_test        =  hyp_dic[str_test_hyp]
    
    ### assigning strings of TRAIN set
    # Initialize & define subjects' features and hypnio list
    X_train       = np.empty((0,np.shape(X_test)[1]))
    y_train       = np.empty((0,np.shape(y_test)[1]))
    
    # Iterate trhough the FEATURES of other subject than the test subject and concatenate them
    for key in subjects_dic:
       if key != str_test_feat:
           feat_tmp = subjects_dic[key]
           X_train  = np.row_stack((X_train, feat_tmp)) 
           del feat_tmp
    # Iterate trhough the LABELS unequal to the test subject and concatenate them       
    for key in hyp_dic:
        if key != str_test_hyp:
            label_tmp = hyp_dic[key]
            y_train  = np.row_stack((y_train, label_tmp)) 
            del label_tmp
    
    # Make a note of the current status:
    print(f'Raw train and test data were created.')
    
    # Replace any probable NaN
    X_train = Object.replace_NaN_with_mean(X_train)
    X_test  = Object.replace_NaN_with_mean(X_test)
    
    # Replace any probable inf
    X_train = Object.replace_inf_with_mean(X_train)
    X_test  = Object.replace_inf_with_mean(X_test)
    
    # Z-score features
    X_train, X_test = Object.Standardadize_features(X_train, X_test)
    
    # Add time dependence to the data classification
    td = 6 # epochs of memory
    X_train_td = Object.add_time_dependence_to_features(X_train, n_time_dependence=td)
    X_test_td  = Object.add_time_dependence_to_features(X_test,  n_time_dependence=td)
    
    # Temporary truncate first and last three epochs
    X_train_td = X_train_td[td:len(X_train_td)-td,:]
    X_test_td  = X_test_td[td:len(X_test_td)-td,:]
    y_train    = y_train[td:len(y_train)-td,:]
    y_test     = y_test[td:len(y_test)-td,:]
    
    # Feature Selection
    y_train_td = Object.binary_to_single_column_label(y_train)
    
    # select features only on first iteration
    if c == 1:
        ranks, Feat_selected, selected_feats_ind = Object.FeatSelect_Boruta(X_train_td,
                                                        y_train_td[:,0], max_depth = 7)
    
    # Apply selected features
    X_train = X_train_td[:, selected_feats_ind]
    X_test  = X_test_td[:, selected_feats_ind]
    
    # Define classifier of interest 
    y_pred = Object.RandomForest_Modelling(X_train, y_train, X_test, y_test, n_estimators = 500)
    
    # Metrics to assess the model performance on test data
    Acc, Recall, prec, f1_sc = Object.multi_label_confusion_matrix(y_test, y_pred)
    
    # Concatenate metrics to store
    all_metrics = [Acc, Recall, prec, f1_sc]
    
    # Store all metrics of the current iteration
    metrics_per_fold["iteration{}".format(c)] = all_metrics
    c = c + 1
    
    # Hypnogram
    hyp_test = Object.binary_to_single_column_label(y_test)
    Object.plot_hyp(hyp = hyp_test, mark_REM = 'active')
    
    # Comparative hypnogram
    hyp_pred = Object.binary_to_single_column_label(y_pred)
    Object.plot_comparative_hyp(hyp_true = hyp_test, hyp_pred = hyp_pred, mark_REM = 'active')
    
    # Save figure
    Object.save_figure(saving_format = '.png',
                       directory = '/project/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.1/Fp1-Fp2/',
                       saving_name = 'test_subject_' + str(c_subj), dpi = 1200,
                       full_screen = False)
    
#%% Save results
path = '/project/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.1/Fp1-Fp2/'
fname = 'LOOCV_results_Fp1-Fp2'
with open(path+fname+'.pickle',"wb") as f:
            pickle.dump(metrics_per_fold, f)    
            
#%% load results
"""
path     = 'P:/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.1/'
filename = 'LOOCV_results'
with open(path + filename + '.pickle', "rb") as f: 
   Outputs = pickle.load(f)
#%% Show final average results
Object.Mean_leaveOneOut(Outputs)

"""