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

#####===================== Reading EDF data files=========================#####

pat_labels = loadtxt("P:/3013080.02/ml_project/patient_labels.txt", delimiter="\t", skiprows = 1)

#####============= Distinguishing patients from control group=============#####

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

# Initialization
subjects_dic     = {}
hyp_dic          = {}
metrics_per_fold = {}

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)

#####============= Iterate through each subject to find data =============#####

for idx, c_subj in enumerate(subj_c):
    print (f'Analyzing Subject Number: {c_subj}')
    ## Read in data
    file     = "P:/3013080.02/ml_project/test_data/LK_" + str(int(c_subj)) + "_1.edf"
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
    Mastoids          = ['TP10', 'TP9', 'TP10', 'TP9', 'TP10', 'TP9', 'TP10', 'TP9'] # Reference electrodes
    RequiredChannels  = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2'] # main electrodes
    # 2. Channels that don't need to be referenced:
    NonRefChannels    = ['ECG']
    Idx               = []
    Idx_Mastoids      = []
    Idx_NonReferenced = []
    
#####================= Find index of required channels ===================#####
    for indx, c in enumerate(AvailableChannels):
        if c in RequiredChannels:
            Idx.append(indx)
        elif c in Mastoids:
            Idx_Mastoids.append(indx)
        elif c in NonRefChannels:
            Idx_NonReferenced.append(indx)

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

    hyp = loadtxt("P:/3013065.04/Depressed_Loreta/hypnograms/LK_" + 
                str(int(c_subj)) + ".txt", delimiter="\t")
    
    ### Create sepereate data subfiles based on hypnogram (N1, N2, N3, NREM, REM) 
    tic      = time.time()
#####================= Concatenation of selected channels ================#####   
    # Calculate referenced channels:
    Idx_Mastoids = [34,106,34,106,34,106,34,106]
    data_epoched_selected = data_epoched[Idx] - data_epoched[Idx_Mastoids]
    data_epoched_nonref = data_epoched[Idx_NonReferenced]
    # Add non-referenced channels: 
    data_epoched_selected_ = np.concatenate([data_epoched_selected, data_epoched_nonref], 0)
    print('Time to split sleep stages per epoch: {}'.format(time.time()-tic))
    
    #%% Analysis section
#####================= remove bad chanbnels and arousals =================#####   
    # assign the proper data and labels
    x_tmp_init = data_epoched_selected_
    y_tmp_init = hyp
    # Remove bad signals (hyp == -1)
    x_tmp, y_tmp =  Object.remove_bad_signals(hypno_labels = y_tmp_init,
                                              input_feats = x_tmp_init)
    
#####============= remove stages contaminated with arousal ===============#####      
    
    x_tmp, y_tmp = Object.remove_arousals(hypno_labels = y_tmp, input_feats = x_tmp)
    # Create binary labels array
    yy = Object.binary_labels_creator(y_tmp)
    # Initialize feature array:
    Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
      
#####================== Extract the relevant features ====================#####    
    for k in np.arange(np.shape(data_epoched_selected_)[0]):
        feat_temp = Object.FeatureExtraction_per_subject(Input_data = x_tmp[k,:,:])
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
   
#%% Save created features and labels
#####====================== Save extracted features ======================#####      
path     = 'F:/Direct_EDF_to_Classification/'
filename = 'sleep_scoring_NoArousal'
Object.save_dictionary(path, filename, hyp_dic, subjects_dic)


#%% Load featureset and labels
path                  = 'F:/Direct_EDF_to_Classification/'
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
    
    # Replace any probable NaN
    X_train = Object.replace_NaN_with_mean(X_train)
    X_test  = Object.replace_NaN_with_mean(X_test)
    
    # Replace any probable inf
    X_train = Object.replace_inf_with_mean(X_train)
    X_test  = Object.replace_inf_with_mean(X_test)
    
    # Z-score features
    X_train, X_test = Object.Standardadize_features(X_train, X_test)
    
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
    hyp_test = Object.create_single_hypno(y_test)
    Object.plot_hyp(hyp = hyp_test, mark_REM = 'active')
    
    # Comparative hypnogram
    hyp_pred = Object.create_single_hypno(y_pred)
    Object.plot_comparative_hyp(hyp_true = hyp_test, hyp_pred = hyp_pred, mark_REM = 'active')
    
# Show final average results
Object.Mean_leaveOneOut(metrics_per_fold)