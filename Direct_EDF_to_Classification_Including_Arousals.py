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
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report, plot_confusion_matrix

#####===================== Defining project folder========================#####

project_folder = "F:/Loreta_data/"
#project_folder = "P:"

#####===================== Reading EDF data files=========================#####

pat_labels = loadtxt(project_folder + "patient_labels.txt", delimiter="\t", skiprows = 1)

#####============= Distinguishing patients from control group=============#####

gp = loadtxt(project_folder + "grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
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
"""
tic_tot = time.time()

# Igonre unnecessary warnings
np.seterr(divide='ignore', invalid='ignore')

#####============= Iterate through each subject to find data =============#####
for idx, c_subj in enumerate(subj_c):

        
    print (f'Analyzing Subject Number: {c_subj}')
    ## Read in data
    file     = project_folder + "data/LK_" + str(int(c_subj)) + "_1.EDF"
    tic      = time.time()
    data     = mne.io.read_raw_edf(file)
    # Data raw EEG --> Deactive
    # data.plot(duration = 30, highpass = .3 , lowpass = 25 )
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

    hyp = loadtxt(project_folder +"hypnograms/LK_" + 
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
#####================= remove chanbnels without scroing ==================#####   
    
    # assign the proper data and labels
    x_tmp_init = data_epoched_selected
    y_tmp_init = hyp
    
    # Ensure equalituy of length for arrays:
    Object.Ensure_data_label_length(x_tmp_init, y_tmp_init)
    
    # Remove non-scored epochs
    x_tmp, y_tmp =  Object.remove_channels_without_scoring(hypno_labels = y_tmp_init,
                                              input_feats = x_tmp_init)
    
    # Remove disconnections
    '''x_tmp, y_tmp =  Object.remove_disconnection(hypno_labels= y_tmp, 
                                                input_feats=x_tmp) '''
    
#####============= Create a one hot encoding form of labels ==============##### 

    # Create binary labels array
    yy = Object.One_hot_encoding(y_tmp)     
    
    # Ensure all the input labels have a class
    Object.Unlabaled_rows_detector(yy)
    
    # Initialize feature array:
    Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
      
#####================== Extract the relevant features ====================#####    
    
    for k in np.arange(np.shape(data_epoched_selected)[0]):
        
        feat_temp         = Object.FeatureExtraction_per_subject(Input_data = x_tmp[k,:,:])
        Feat_all_channels = np.column_stack((Feat_all_channels,feat_temp))
        
    toc = time.time()
    print(f'Features of subject {c_subj} were successfully extracted in: {toc-tic} secs')
    
    # Double check the equality of size of arrays
    Object.Ensure_feature_label_length(Feat_all_channels, yy)
    
    # Defining dictionary to save features PER SUBJECT
    subjects_dic["subject{}".format(c_subj)] = Feat_all_channels
    
    # Defining dictionary to save hypnogram PER SUBJECT
    hyp_dic["hyp{}".format(c_subj)] = yy
    
#####=============== Removing variables for next iteration ===============#####      
    del x_tmp, y_tmp, feat_temp, yy
    toc = time.time()
    
    print('Feature extraction of subject {c_subj} has been finished.')   

print('Total feature extraction of subjects took {tic_tot - time.time()} secs.')
#%% Save created features and labels

#####====================== Save extracted features ======================#####      

path     = project_folder +"features/"
filename = 'sleep_scoring_Fp1-Fp2_220520_IncludeContaminatedStagesWithArtefact_IncludeDisconnection_RemoveOnly-1_ExcludingSubject35'
Object.save_dictionary(path, filename, hyp_dic, subjects_dic)

"""

#%% Load featureset and labels

path                  =  project_folder + "features/"
filename              =  "sleep_scoring_NoArousal_8channels"
#filename              = "sleep_scoring_NoArousal_8channels"
subjects_dic, hyp_dic = Object.load_dictionary(path, filename)

#%% ================================Training part==============================

# Training perentage
train_size = .8
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
tic                = time.time()
test_subjects_list = []
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
    
    # keep the subject id
    test_subjects_list.append(str_test_feat)
    
    # remove for next iteration
    del tmp_x, tmp_y, str_test_feat, str_test_hyp
    
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

########========== select features only on first iteration ============########

td = 6 # Time dependence: number of epochs of memory

X_train_td = Object.add_time_dependence_backward(X_train, n_time_dependence=td,
                                                    padding_type = 'sequential')

X_test_td  = Object.add_time_dependence_backward(X_test,  n_time_dependence=td,
                                                    padding_type = 'sequential')

########====================== Feature Selection ======================########

y_train_td = Object.binary_to_single_column_label(y_train)

########========== select features only on first iteration ============########
"""
ranks, Feat_selected, selected_feats_ind = Object.FeatSelect_Boruta(X_train_td,
                                                    y_train_td[:,0], max_iter = 50, max_depth = 7)

#######===================== Save selected feats =======================#######

path                  =  project_folder + "features/"
filename              = "Selected_Features_BoturaAfterTD=6_EightChannels_230520_Backward"
import pickle        
with open(path+filename+'.pickle',"wb") as f:
    pickle.dump(selected_feats_ind, f)
"""
########################### Load selected feats ###############################

path                  =  project_folder  + "features/"
filename              = "sleep_scoring_NoArousal_8channels_selected_feats_NEW"
#filename              = "sleep_scoring_NoArousal_8channels_selected_feats_NEW"
with open(path + filename + '.pickle', "rb") as f: 
    selected_feats_ind = pickle.load(f)
    
########=================== Apply selected features ===================########

X_train = X_train_td[:, selected_feats_ind]
X_test  = X_test_td[:, selected_feats_ind]

########============== Define classifier of interest ==================########

y_pred = Object.KernelSVM_Modelling(X_train, y_train,X_test, y_test, kernel='rbf')
y_pred = Object.XGB_Modelling(X_train, y_train,X_test, y_test, n_estimators = 300, 
                      max_depth=3, learning_rate=.1)

########===== Metrics to assess the model performance on test data ====########

Acc, Recall, prec, f1_sc, kappa, mcm= Object.multi_label_confusion_matrix(y_test, y_pred)

########================= Creating subjective outputs =================########

Object.create_subjecive_results(y_true=y_test, y_pred=y_pred, 
                                test_subjects_list = test_subjects_list,
                                subjects_data_dic = subjects_dic,
                                fname_save = "results")

########============= find number of epochs per stage =================########

Object.find_number_of_samples_per_class(y_test, including_artefact = False)

########================== Comparative hypnogram ======================########

hyp_test = Object.binary_to_single_column_label(y_test)
hyp_pred = Object.binary_to_single_column_label(y_pred)
Object.plot_comparative_hyp(hyp_true = hyp_test, hyp_pred = hyp_pred, mark_REM = 'active')

########==================== Plot subjectve hypnos ====================########

Object.plot_subjective_hypno(y_true=y_test, y_pred=y_pred, 
                             test_subjects_list=test_subjects_list,
                             subjects_data_dic=subjects_dic,
                             save_fig = True, 
                             directory="C:/PhD/Github/ssccoorriinngg/")

########================== Plot subjective conf-mat  ==================########

Object.plot_confusion_mat_subjective(y_true=y_test, y_pred=y_pred, 
                             test_subjects_list=test_subjects_list,
                             subjects_data_dic=subjects_dic)

########========================== Save figure =======================#########
Object.save_figure(saving_format = '.png',
                   directory = '/project/3013080.02/Mahdad/Github/ssccoorriinngg/Plots/v0.2/Fp1-Fp2/',
                   saving_name = 'test_subject_all' + str(c_subj), dpi = 900,
                   full_screen = False)

