# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:07:37 2020

CopyRight: Mahdad Jafarzadeh 2020
    
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import tensorflow as tf
from scipy import signal
%matplotlib qt

#####==================== Defining required paths r=======================#####

# Use this for local pc
main_path = "D:/Loreta_data/"
data_path = main_path + "/data/"

# Use this for clusetr
main_path = "/project/3013080.02/Loreta_data/"
data_path = main_path + "/data/"

#####============= Distinguishing patients from control group=============#####

gp = loadtxt(main_path + "grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients

# Just keep controls 
for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))


# Initialization
subjects_dic     = {}
hyp_dic          = {}
metrics_per_fold = {}
fs_dic = {}
#####============= Iterate through each subject to find data =============#####
# =============================================================================
# Tic = time.time()
# 
# for idx, c_subj in enumerate(subj_c):
#     
#     print (f'Analyzing Subject Number: {c_subj}')
#     
#     ## Read in data
#     file     = data_path + "LK_" + str(int(c_subj)) + "_1.edf"
#     tic      = time.time()
#     raw      = mne.io.read_raw_edf(file, preload = True)
# 
#     print('Time to read EDF: {}'.format(time.time()-tic))
#     
# #####=================Retrieving information from data====================#####
#     
#     DataInfo          = raw.info
#     AvailableChannels = DataInfo['ch_names']
#     fs                = int(DataInfo['sfreq'])
#     print(f'Sampling freq: {fs}')
#     
# #####============= create an object of ssccoorriinngg class ==============#####
#     
#     # fs is defined here to account any inconsistency in fs among recordings
#     Object = ssccoorriinngg(filename='', channel='', fs = fs, T = 30)
#     
# ####============================FIR Filtering =============================####
#     
#     f_min = .1 #Hz
#     f_max = 30 #Hz
#     tic   = time.time()
#     raw.filter(l_freq=f_min, h_freq=f_max)
#     print('Filtering time: {}'.format(time.time()-tic))
#     
#     # Get filtered data
#     filtered_data = raw.get_data()
#     
# #####==================Choosing channels of interest======================#####
#     
#     References       = ['Fp2'] # M2 and M1, respectively
#     RequiredChannels = ['Fp1'] 
#     
# #####================= Find index of required channels ===================#####
#     
#     # Initializing index lists
#     Idx = []
#     Idx_Mastoids = []
#     
#     # Find index of required channels     
#     for indx, c in enumerate(RequiredChannels):
#         if c in AvailableChannels:
#             Idx.append(AvailableChannels.index(c))
#             
#     # Find index of refernces (e.g. Mastoids) 
#     for indx, c in enumerate(References):
#         if c in AvailableChannels:
#             Idx_Mastoids.append(AvailableChannels.index(c))
# 
# #####===== Sampling rate is 200hz; thus 1 epoch(30s) is 6000 samples =====#####
#             
#     T = 30 #secs
#     len_epoch   = fs * T
#     start_epoch = 0
#     n_channels  =  len(AvailableChannels)
#        
# #####============ Cut tail; use modulo to find full epochs ===============#####
# 
#     filtered_data = filtered_data[:, 0:filtered_data.shape[1] - filtered_data.shape[1]%len_epoch]
#     
# #####========== Reshape data [n_channel, len_epoch, n_epochs] ============#####
#     
#     data_epoched = np.reshape(filtered_data,
#                               (n_channels, len_epoch,
#                                int(filtered_data.shape[1]/len_epoch)), order='F' )
#     
# #####===================== Reading hypnogram data ========================#####
# 
#     # Select channels of interest
#     data_epoched_selected = data_epoched[Idx,:,:] - data_epoched[Idx_Mastoids,:,:]
#     
#     ## Read Hypnogram
#     hyp = loadtxt(main_path + "hypnograms/LK_" +
#                  str(int(c_subj)) + ".txt", delimiter="\t")
# 
#     #####================= Find order of the selected channels ===============#####  
#  
#     #Init
#     picked_channels = []
#     picked_refs     = []
#     List_Channels   = []
#     
#     # Find main channels
#     for jj,kk in enumerate(Idx):
#         picked_channels = np.append(picked_channels, AvailableChannels[kk])
#     
#     # Show subject ID 
#     print(f'subject LK {c_subj} ... Picked channels:')
#     
#     # Find references
#     for jj,kk in enumerate(Idx_Mastoids):
#         picked_refs     = np.append(picked_refs, AvailableChannels[kk])
#         print(f'{str(picked_channels[jj])}-{str(picked_refs[jj])}')
#     
#     # Create lis of channels
#     for kk in np.arange(0, len(Idx)):
#         List_Channels = np.append(List_Channels, picked_channels[kk] + '-' + picked_refs[kk])
#         
#     #%% Analysis section
# #####================ remove bad chanbnels and artefacts =================#####   
#     
#     # assign the proper data and labels
#     x_tmp_init = data_epoched_selected
#     y_tmp_init = hyp
# 
#     # Ensure equalituy of length for arrays:
#     Object.Ensure_data_label_length(x_tmp_init, y_tmp_init)
#     
#     # Remove disconnections
#     x_tmp, y_tmp =  Object.remove_bad_signals(hypno_labels= y_tmp_init, 
#                                                 input_feats=x_tmp_init) 
#     
#     # Remove artefactual epochs
#     x_tmp, y_tmp = Object.remove_artefact(hypno_labels = y_tmp, input_feats = x_tmp)
#     
# #####================== Extract the relevant features ====================#####    
#     
#     # Create binary labels array
#     yy = Object.binary_labels_creator_categories(y_tmp)
# 
#     # Initialize feature array:
#     Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
#       
#     # Extract features
#     for k in np.arange(np.shape(data_epoched_selected)[0]):
#         
#         feat_temp = Object.FeatureExtraction_per_subject(Input_data = x_tmp[k,:,:], fs = fs)
#         
#         # Concatenate features
#         Feat_all_channels = np.column_stack((Feat_all_channels,feat_temp))
# 
#     print(f'Features of subject {c_subj} were successfully extracted in: {time.time()-tic} secs')
#     
# ####================ Store subjective features and labels ================ ####    
#     
#     # Defining dictionary to save features PER SUBJECT
#     subjects_dic["subject{}".format(c_subj)] = Feat_all_channels
#     
#     # Defining dictionary to save hypnogram PER SUBJECT
#     hyp_dic["hyp{}".format(c_subj)] = yy
#     
#     fs_dic["fs_{}".format(c_subj)] = fs
#     
#     print(f'Features and hypno of subject {c_subj} were successfully added to dictionary')
#     print('Feature extraction of subject {c_subj} has been finished.')   
#     
# #####=============== Removing variables for next iteration ===============##### 
#      
#     del x_tmp, y_tmp, feat_temp, yy
#     toc = time.time()
# 
# # Report the end of feature extraction
# print('Total feature extraction of subjects took {Tic - time.time()} secs.')
# 
# #%% Save created features and labels
# #####====================== Save extracted features ======================#####      
# 
# path     = main_path  
# filename = "Scoring_Features_Loreta_Fp1-Fp2_061020"
# Object.save_dictionary(path, filename, hyp_dic, subjects_dic)
# 
# =============================================================================
#%% Load featureset and labels
path                  = main_path  
filename              = "Scoring_Features_Loreta_Fp1-M2_Fp2-M1_041020"
subjects_dic, hyp_dic = Object.load_dictionary(path, filename)

#%% Create onbject of ssccoorriinngg
Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)

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
    X_train_td = Object.add_time_dependence_backward(X_train, n_time_dependence=td,\
                                                     padding_type = 'sequential')
    X_test_td  = Object.add_time_dependence_backward(X_test,  n_time_dependence=td,\
                                                     padding_type = 'sequential')
    
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
    y_pred = Object.ANN_classifier(X_train, y_train, X_test, units_h1=1000, units_h2 = 500, units_output = 5,
                              activation_out = 'softmax',
                              init = 'uniform', activation = 'relu', optimizer = 'adam',
                              loss = 'categorical_crossentropy', metrics=[tf.keras.metrics.Recall()],
                              h3_status = 'deactive', units_h3 = 50, epochs = 100, batch_size = 100)
    
    # Metrics to assess the model performance on test data
    Acc, Recall, prec, f1_sc, kappa, mcm= Object.multi_label_confusion_matrix(y_test, y_pred)
    
    # Concatenate metrics to store
    all_metrics = [Acc, Recall, prec, f1_sc, kappa]
    
    # Store all metrics of the current iteration
    metrics_per_fold["iteration{}".format(c)] = all_metrics
    c = c + 1
    
    # Hypnogram
# =============================================================================
#     hyp_test = Object.binary_to_single_column_label(y_test)
#     Object.plot_hyp(hyp = hyp_test, mark_REM = 'active')
# =============================================================================
    
    # Comparative hypnogram
    hyp_true = Object.binary_to_single_column_label(y_test)  
    Object.plot_comparative_hyp(hyp_true = hyp_true, hyp_pred = y_pred, mark_REM = 'active')
    
    # Save figure 
    Object.save_figure(saving_format = '.png',
                   directory= main_path + "LOOCV/hypnos/",
                   saving_name = 'test_hyp_' + str(c_subj), dpi = 900,
                   full_screen = False)

    del y_pred, y_test, X_train, X_test, hyp_true
#%% Save results
path = main_path
fname = 'LOOCV_results_Fp1-Fp2_Loreta_ANN'
with open(path+fname+'.pickle',"wb") as f:
            pickle.dump(metrics_per_fold, f)    
            
#%% load results
# =============================================================================
# path     = 'P:/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.1/'
# filename = 'LOOCV_results'
# with open(path + filename + '.pickle', "rb") as f: 
#    Outputs = pickle.load(f)
# =============================================================================
            
#%% Show final average results
Object.Mean_leaveOneOut(metrics_per_fold)



