# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:39:54 2020

@author: mahjaf

This is the main file to utilize "ssccoorriinngg" class fot the purpose of 
automatic sleep staging This is an example file how to use the class, meaning
that all the functions and capabilitiesof class have not been used.
For mor details see the class itself: "ssccoorriinngg.py"
"""
#%% Importing libs
import numpy as np
from   numpy import loadtxt
import h5py
import time
import os 
from ssccoorriinngg import ssccoorriinngg

#%% Distinguishing patients from control group
gp = loadtxt("P:/3013080.02/ml_project/grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients
# Find control subject IDs
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
# create an object of ssccoorriinngg class
Object = ssccoorriinngg(filename='', channel='fp1-fp2', fs = 200, T = 30)

#%% Read data per subject and assign it to relevant array
for idx, c_subj in enumerate(subj_c):
    print (f'Analyzing Subject Number: {c_subj}')
    tic = time.time()
    path = 'D:/1D_TimeSeries/raw_EEG/full/Fp1-Fp2/'
    with h5py.File(path +'LK_'+ str(c_subj) + '_1.h5', 'r') as rf:
        x_tmp_init  = rf['.']['data_fp1-fp2'].value
        y_tmp_init  = rf['.']['hypnogram'].value
    print (f'Featrueset and hypno of subject {c_subj} was successfully loaded.')
    
    # Remove bad signals (hyp == -1)
    x_tmp, y_tmp =  Object.remove_bad_signals(hypno_labels = y_tmp_init,
                                              input_feats = x_tmp_init)
    
    # remove stages contaminated with arousal
    x_tmp, y_tmp = Object.remove_arousals(hypno_labels = y_tmp, input_feats = x_tmp)

    # Create binary labels array
    yy = Object.binary_labels_creator(y_tmp)
    
    # Extract the relevant features
    feat_temp = Object.FeatureExtraction_per_subject(Input_data = x_tmp)
    toc = time.time()
    print(f'Features of subject {c_subj} were successfully extracted in: {toc-tic} secs')
    ############### ALSO IRRELAVNT LABELS SHOULD BE REMOVED
    
    # Defining dictionary to save features PER SUBJECT
    subjects_dic["subject{}".format(c_subj)] = feat_temp
    
    # Defining dictionary to save hypnogram PER SUBJECT
    hyp_dic["hyp{}".format(c_subj)] = yy
    
    # removing variables for next iteration
    del x_tmp, y_tmp, feat_temp, yy
    toc = time.time()
    print(f'Features and hypno of subject {c_subj} were successfully added to dictionary')

print('Feature extraction has been finished.')   
 
#%% Save created features and labels
path     = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
filename = 'sleep_scoring_NoArousal'
Object.save_dictionary(path, filename, hyp_dic, subjects_dic)

#%% Load featureset and labels
path                  = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
filename              = 'sleep_scoring_NoArousal'
subjects_dic, hyp_dic = Object.load_dictionary(path, filename)

#%% Create leave-one-out cross-validation
 # Define counter of metrics per fold
c = 1
for idx, c_subj in enumerate(subj_c):
   
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
    X_train       = np.empty((0,42))
    y_train       = np.empty((0,5))
    
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
    
    # Z-score features
    X_train, X_test = Object.Standardadize_features(X_train, X_test)
    
    # Replace any probable NaN
    X_train = Object.replace_NaN_with_mean(X_train)
    X_test  = Object.replace_NaN_with_mean(X_test)
    
    # Define classifier of interest
    y_pred = Object.RandomForest_Modelling(X_train, y_train, X_test, y_test, n_estimators = 1000)
    
    # Metrics to assess the model performance on test data
    Acc, Recall, prec, f1_sc = Object.multi_label_confusion_matrix(y_test, y_pred)
    
    # Concatenate metrics to store
    all_metrics = [Acc, Recall, prec, f1_sc]
    
    # Store all metrics of the current iteration
    metrics_per_fold["iteration{}".format(c)] = all_metrics
    c = c + 1
    
    # Hypnogram
    #hyp_test = Object.create_single_hypno(y_test)
    #Object.plot_hyp(hyp = hyp_test, mark_REM = 'active')
    
    
