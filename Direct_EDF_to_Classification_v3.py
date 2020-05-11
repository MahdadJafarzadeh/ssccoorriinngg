# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:44:03 2020

@author: mahjaf
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score

#####===================== Defining project folder========================#####

project_folder = "/project"
#project_folder = "P:"

#####===================== Reading EDF data files=========================#####

pat_labels = loadtxt(project_folder + "/3013080.02/ml_project/patient_labels.txt", delimiter="\t", skiprows = 1)

#####============= Distinguishing patients from control group=============#####

gp = loadtxt(project_folder + "/3013080.02/ml_project/grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
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

# Igonre unnecessary warnings
np.seterr(divide='ignore', invalid='ignore')

#####============= Iterate through each subject to find data =============#####

for idx, c_subj in enumerate(subj_c):
    print (f'Analyzing Subject Number: {c_subj}')
    ## Read in data
    file     = project_folder + "/3013080.02/ml_project/test_data/LK_" + str(int(c_subj)) + "_1.EDF"
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

    hyp = loadtxt(project_folder +"/3013065.04/Depressed_Loreta/hypnograms/LK_" + 
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
    
    x_tmp, y_tmp = Object.remove_arousals(hypno_labels = y_tmp, input_feats = x_tmp)
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
    
    #####============================= Save per case =========================#####
    path     = project_folder +"/3013080.02/ml_project/scripts/1D_TimeSeries/features/percase"
    filename = "/LK__"+ str(int(c_subj))
    Object.save_dictionary(path, filename, hyp_dic, subjects_dic)
    
#####=============== Removing variables for next iteration ===============#####      
    del x_tmp, y_tmp, feat_temp, yy
    toc = time.time()
    print(f'Features and hypno of subject {c_subj} were successfully added to dictionary')
    
    print('Feature extraction of subject {c_subj} has been finished.')   

print('Total feature extraction of subjects took {tic_tot - time.time()} secs.')
#%% Save created features and labels


#####====================== Save extracted features ======================#####      

path     = project_folder +"/3013080.02/ml_project/scripts/1D_TimeSeries/features/"
filename = 'sleep_scoring_NoArousal_Fp1-Fp2_full_'
Object.save_dictionary(path, filename, hyp_dic, subjects_dic)


#%% Load featureset and labels
"""
path                  =  project_folder +"/3013080.02/ml_project/scripts/1D_TimeSeries/features/"
filename              = "sleep_scoring_NoArousal_Fp1-Fp2_080520"
subjects_dic, hyp_dic = Object.load_dictionary(path, filename)

#%% ================================Training part==============================

# Training perentage
train_size = .75
n_train = round(train_size * len(subj_c))

#######=== Randomly shuffle subjects to choose train and test splits ===#######

subj_c = np.random.RandomState(seed=42).permutation(subj_c)

#######=============== Initialize train and test arrays ================#######

X_train = np.empty((0, np.shape(subjects_dic['subject14'])[1]))
X_test  = np.empty((0, np.shape(subjects_dic['subject14'])[1]))
y_train = np.empty((0, np.shape(hyp_dic['hyp14'])[1]))
y_test  = np.empty((0, np.shape(hyp_dic['hyp14'])[1]))

########======= Picking the train subjetcs and concatenate them =======########
tic = time.time()

for c_subj in subj_c[0:n_train]:
    
    # train hypnogram
    str_train_hyp  = 'hyp' + str(c_subj)
    
    # train featureset
    str_train_feat = 'subject' + str(c_subj)
    
    # create template arrays for featurs and label
    tmp_x          =  subjects_dic[str_train_feat]
    tmp_y          =  hyp_dic[str_train_hyp]
    
    # Concatenate features and labels
    X_train = np.row_stack((X_train, tmp_x))
    y_train = np.row_stack((y_train, tmp_y))
    del tmp_x, tmp_y
print('Training set was successfully created in : {} secs'.format(time.time()-tic))

#%% ================================Test part==============================%%#

########======== Picking the test subjetcs and concatenate them =======########
tic = time.time()

for c_subj in subj_c[n_train:]:
    
    # test hypnogram
    str_test_hyp  = 'hyp' + str(c_subj)
    
    # test featureset
    str_test_feat = 'subject' + str(c_subj)
    
    # create template arrays for featurs and  label
    tmp_x         =  subjects_dic[str_test_feat]
    tmp_y         =  hyp_dic[str_test_hyp]
    
    # Concatenate features and labels
    X_test = np.row_stack((X_test, tmp_x))
    y_test = np.row_stack((y_test, tmp_y))
    del tmp_x, tmp_y
print('Test set was successfully created in : {} secs'.format(time.time()-tic))

print(f'Raw train and test data were created.')

########================== Replace any probable NaN ===================########

X_train = Object.replace_NaN_with_mean(X_train)
X_test  = Object.replace_NaN_with_mean(X_test)

########================== Replace any probable inf ===================########

X_train = Object.replace_inf_with_mean(X_train)
X_test  = Object.replace_inf_with_mean(X_test)

########==================== Z-score of features ======================########

X_train, X_test = Object.Standardadize_features(X_train, X_test)

########======= Add time dependence to the data classification ========########

td = 6 # Time dependence: # epochs of memory
X_train_td = Object.add_time_dependence_to_features(X_train, n_time_dependence=td)
X_test_td  = Object.add_time_dependence_to_features(X_test,  n_time_dependence=td)

########======== Temporary truncate first and last "td" epochs ========########

X_train_td = X_train_td[td:len(X_train_td)-td,:]
X_test_td  = X_test_td[td:len(X_test_td)-td,:]
y_train    = y_train[td:len(y_train)-td,:]
y_test     = y_test[td:len(y_test)-td,:]

########====================== Feature Selection ======================########

y_train_td = Object.binary_to_single_column_label(y_train)

########========== select features only on first iteration ============########

ranks, Feat_selected, selected_feats_ind = Object.FeatSelect_Boruta(X_train_td,
                                                    y_train_td[:,0], max_depth = 7)

########=================== Apply selected features ===================########

X_train = X_train_td[:, selected_feats_ind]
X_test  = X_test_td[:, selected_feats_ind]

########============== Define classifier of interest ==================########

y_pred = Object.RandomForest_Modelling(X_train, y_train, X_test, y_test, n_estimators = 500)

########===== Metrics to assess the model performance on test data ====########

Acc, Recall, prec, f1_sc = Object.multi_label_confusion_matrix(y_test, y_pred)

########=============== Concatenate metrics to store ==================########

all_metrics = [Acc, Recall, prec, f1_sc]

########========================== Hypnogram ==========================########

hyp_test = Object.binary_to_single_column_label(y_test)
Object.plot_hyp(hyp = hyp_test, mark_REM = 'active')

########================== Comparative hypnogram ======================########

hyp_pred = Object.binary_to_single_column_label(y_pred)
Object.plot_comparative_hyp(hyp_true = hyp_test, hyp_pred = hyp_pred, mark_REM = 'active')

########========================== Save figure =======================#########
Object.save_figure(saving_format = '.png',
                   directory = '/project/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.2/Fp1-Fp2/',
                   saving_name = 'test_subject_all' + str(c_subj), dpi = 900,
                   full_screen = False)

#########======================= Randomized search ===================#########
# Define sccoring metrics
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}   
# Apply randomized search
Object.RandomSearchRF(X = X_train, y = y_train, scoring = scoring, estimator = RandomForestClassifier(),
                        n_estimators = [int(x) for x in np.arange(10, 1000, 50)],
                        max_features = ['log2', 'sqrt'],
                        max_depth = [int(x) for x in np.arange(10, 100, 30)],
                        min_samples_split = [2, 5, 10],
                        min_samples_leaf = [1, 2, 4],
                        bootstrap = [True, False],
                        n_iter = 100, cv = 10)
#%% Save results
path = '/project/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.1/Fp1-Fp2/'
fname = 'LOOCV_results_Fp1-Fp2'
with open(path+fname+'.pickle',"wb") as f:
            pickle.dump(metrics_per_fold, f)    
            
#%% load results

path     = 'P:/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.1/'
filename = 'LOOCV_results'
with open(path + filename + '.pickle', "rb") as f: 
   Outputs = pickle.load(f)
#%% Show final average results
Object.Mean_leaveOneOut(Outputs)
"""
