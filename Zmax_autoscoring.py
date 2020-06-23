# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:41:54 2020

@author: mahjaf

Automatic sleep scoring implemented for Zmax headband.

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
#####==================== Defining required paths r=======================#####

Main_path         = "P:/3013080.01/"
subject_Id_folder = Main_path + "Autoscoring/ssccoorriinngg/"
Data_folder       = Main_path + "Zmax_Data/"
Hypnogram_folder  = Main_path + "somno_scorings/Rathiga/"

#####===================== Reading EDF data files=========================#####

subject_ids = loadtxt(subject_Id_folder+"Zmax/Subject_ids_excluding 22_2.txt", dtype = 'str',delimiter='\n')

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 256, T = 30)
# =============================================================================
# 
# #Initialization
# subjects_dic     = {}
# hyp_dic          = {}
# metrics_per_fold = {}
# raw_data_dic     = {}
# tic_tot = time.time()
# 
# # Igonre unnecessary warnings
# np.seterr(divide='ignore', invalid='ignore')
# 
# #####============= Iterate through each subject to find data =============#####
# 
# for idx, c_subj in enumerate(subject_ids):
# 
#     # Report the investigational file  
#     print (f'Analyzing Subject Number: {c_subj[0:11]}')
#     
#     # Separate subject ID and corresponding night
#     subj_night_folder = c_subj[0:4] + "/" + c_subj[5:11] + "/"
#     
#     ## Read in data
#     tic      = time.time()
#     file     = Data_folder + subj_night_folder
#     
#     # Reading EEG left and right
#     data_L     = mne.io.read_raw_edf(file + "EEG L.edf")
#     data_R     = mne.io.read_raw_edf(file + "EEG R.edf")
#     
#     # Data raw EEG --> Deactive
#     # data_L.plot(duration = 30, highpass = .3 , lowpass = 25 )
#     
#     # Get EEG data from left and right channels
#     raw_data_L = data_L.get_data()
#     raw_data_R = data_R.get_data()
#     
#     # Combine channels
#     raw_data = np.concatenate((raw_data_L, raw_data_R))
#     
#     print('Time to read EDF: {}'.format(time.time()-tic))
#     
# #####=================== Reading acceleration data =======================#####
#     # Read Acc per axis and also norm of Acc
#     AccNorm, Acc = Object.Read_Acceleration_data(folder_acc = Data_folder + subj_night_folder,
#                                                  axis_files = ["dX", "dY", "dZ"],
#                                                  file_format = ".edf", plot_Acc = False)
# #####=================== Feature extraction Acc =======================#####
# 
#     Feats_Acc = Object.Acc_feature_extraction(AccNorm, Acc, fs=256, axes_acc_status = 'deactive')
#     
# #####=================Retrieving information from data====================#####
#     
# # =============================================================================
# #     DataInfo          = raw_data_L.info
# #     AvailableChannels = DataInfo['ch_names']
# #     fs                = int(DataInfo['sfreq'])
# # =============================================================================
#     
# #####===== Sampling rate is 256Hz; thus 1 epoch(30s) is 6000 samples =====#####
#     
#     fs = 256 #Hz       
#     T  = 30 #secs
#     len_epoch   = fs * T
#     n_channels  = 2  
#     
# #####============ Cut tail; use modulo to find full epochs ===============#####
#     
#     raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
#     
# #####========== Reshape data [n_channel, len_epoch, n_epochs] ============#####
#     
#     data_epoched = np.reshape(raw_data,
#                              (n_channels, len_epoch,
#                              int(raw_data.shape[1]/len_epoch)), order='F' )
#     
# #####===================== Reading hypnogram data ========================#####
#     
#     hyp = loadtxt(subject_Id_folder+"Zmax/"+c_subj+".txt")
#     
#     #Object.find_unscored(hyp, subject_no =c_subj)
#     
#     #%% Analysis section
# 
#     # assign the proper data and labels
#     x_tmp_init = data_epoched
#     y_tmp_init = hyp
#     #####=============== Create a new channel (ch1 - ch2) ================##### 
# 
#     new_ch = x_tmp_init[0] - x_tmp_init[1]
#     new_ch = new_ch[np.newaxis,:,:]
#     
#     # add new channel to channel list
#     #x_tmp_init = np.row_stack((x_tmp_init, new_ch))
#     x_tmp_init = new_ch  # pick only ch1-ch2
# 
#     #####================ remove chanbnels without scroing ===============##### 
#     # Ensure equalituy of length for arrays:
#     Object.Ensure_data_label_length(x_tmp_init, y_tmp_init)
#     
#     # Remove non-scored epochs
#     x_tmp, y_tmp, Feats_Acc =  Object.remove_channels_without_scoring(hypno_labels = y_tmp_init,
#                                               input_feats = x_tmp_init, Feats_Acc=Feats_Acc, 
#                                               Acc_feats = True)
#     
#     # Remove disconnections
#     x_tmp, y_tmp, Feats_Acc =  Object.remove_disconnection(hypno_labels= y_tmp, 
#                                                 input_feats=x_tmp, Feats_Acc=Feats_Acc,
#                                                 Acc_feats = True) 
# 
# #####============= Create a one hot encoding form of labels ==============##### 
# 
#     # Create binary labels array
#     yy = Object.One_hot_encoding(y_tmp)     
#     
#     # Ensure all the input labels have a class
#     Object.Unlabaled_rows_detector(yy)
#     
#     # Initialize feature array:
#     Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
#       
# #####================== Extract the relevant features ====================#####    
#     print(f'Extracting features of subject {c_subj} ....')
#     
#     for k in np.arange(np.shape(x_tmp)[0]):
#         
#         feat_temp         = Object.FeatureExtraction_per_subject(Input_data = x_tmp[k,:,:])
#         Feat_all_channels = np.column_stack((Feat_all_channels,feat_temp))
#     
#     # Adding Acc feats to EEG feats
#     Feat_all_channels = np.column_stack((Feat_all_channels,Feats_Acc))
#     
#     toc = time.time()
#     print(f'Features of subject {c_subj} were successfully extracted in: {toc-tic} secs')
#     
#     # Double check the equality of size of arrays
#     Object.Ensure_feature_label_length(Feat_all_channels, yy)
#     
#     # Defining dictionary to save features PER SUBJECT
#     subjects_dic["subject{}".format(c_subj)] = Feat_all_channels
#     
#     # Defining dictionary to save hypnogram PER SUBJECT
#     hyp_dic["hyp{}".format(c_subj)] = yy
#     
#     # Defining dictionary to save EEG raw data PER SUBJECT
#     raw_data_dic["subject{}".format(c_subj)] = x_tmp
#     
#     
# #####=============== Removing variables for next iteration ===============#####      
#     del x_tmp, y_tmp, feat_temp, yy, Feats_Acc
#     toc = time.time()
#     
#     print('Feature extraction of subject {c_subj} has been finished.')   
# 
# print('Total feature extraction of subjects took {tic_tot - time.time()} secs.')
# #%% Save created features and labels
# 
# #####====================== Save extracted features ======================#####      
# 
# path     = subject_Id_folder +"features/"
# filename = "Zmax_Rathiga_scorings_ch-ch2+AccFeats_190620"
# Object.save_dictionary(path, filename, hyp_dic, subjects_dic)
# 
# #####====================== Save raw data as pickle ======================#####      
# 
# path     =subject_Id_folder +"features/"
# 
# filename = "Zmax_Rathiga_scorings_RawData_ch1-ch2+AccFeats_190620"
# Object.save_dictionary(path, filename, hyp_dic, raw_data_dic)
# =============================================================================

#%% Load featureset and labels

path     = subject_Id_folder +"features/"
filename              =  "Zmax_Rathiga_scorings_ch-ch2+AccFeats_190620"
subjects_dic, hyp_dic = Object.load_dictionary(path, filename)
  
#%% ================================Training part==============================

# Training perentage
train_size = .7
n_train = round(train_size * len(subject_ids))

#######=== Randomly shuffle subjects to choose train and test splits ===#######

subject_ids = np.random.RandomState(seed=0).permutation(subject_ids)

#######=============== Initialize train and test arrays ================#######
sample_subject = "subjectP_12_night1_scoring.csv.spisop.new - Copy"
sample_hyp     = "hypP_12_night1_scoring.csv.spisop.new - Copy" 
X_train = np.empty((0, np.shape(subjects_dic[sample_subject])[1]))
X_test  = np.empty((0, np.shape(subjects_dic[sample_subject])[1]))
y_train = np.empty((0, np.shape(hyp_dic[sample_hyp])[1]))
y_test  = np.empty((0, np.shape(hyp_dic[sample_hyp])[1]))

########======= Picking the train subjetcs and concatenate them =======########
tic = time.time()
train_subjects_list = []
for c_subj in subject_ids[0:n_train]:
    
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
    
    # Keep the train subject
    train_subjects_list.append(str_train_feat)
    del tmp_x, tmp_y
    
print('Training set was successfully created in : {} secs'.format(time.time()-tic))

#%% ================================Test part==============================%%#

########======== Picking the test subjetcs and concatenate them =======########
tic                = time.time()
test_subjects_list = []
for c_subj in subject_ids[n_train:]:
   
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

td = 5 # Time dependence: number of epochs of memory

X_train_td = Object.add_time_dependence_backward(X_train, n_time_dependence=td,
                                                    padding_type = 'sequential')

X_test_td  = Object.add_time_dependence_backward(X_test,  n_time_dependence=td,
                                                    padding_type = 'sequential')

########====================== Feature Selection ======================########

y_train_td = Object.binary_to_single_column_label(y_train)

########========== select features only on first iteration ============########

# =============================================================================
# ranks, Feat_selected, selected_feats_ind = Object.FeatSelect_Boruta(X_train_td,
#                                                     y_train_td[:,0], max_iter = 50, max_depth = 7)
# 
# #######===================== Save selected feats =======================#######
# 
# path     = subject_Id_folder +"features/"
# filename              = "Selected_Features_BoturaAfterTD=5_Backward_Zmax_ch1-ch2+Acc_200620"
# with open(path+filename+'.pickle',"wb") as f:
#     pickle.dump(selected_feats_ind, f)
# =============================================================================
     
########################### Load selected feats ###############################

path     = subject_Id_folder +"features/"
filename              = "Selected_Features_BoturaAfterTD=5_Backward_Zmax_ch1-ch2+Acc_200620"
#filename              = "sleep_scoring_NoArousal_8channels_selected_feats_NEW"
with open(path + filename + '.pickle', "rb") as f: 
    selected_feats_ind = pickle.load(f)
    
########=================== Apply selected features ===================########

X_train = X_train_td[:, selected_feats_ind]
X_test  = X_test_td[:, selected_feats_ind]

########============== Define classifier of interest ==================########
#y_pred = Object.XGB_Modelling(X_train, y_train,X_test, y_test, n_estimators = 500)
#y_pred = Object.KernelSVM_Modelling(X_train, y_train,X_test, y_test, kernel='rbf')
y_pred = Object.ANN_classifier(X_train, y_train, X_test, units_h1=600, units_h2 = 300, units_output = 5,
                              activation_out = 'softmax',
                              init = 'uniform', activation = 'relu', optimizer = 'adam',
                              loss = 'categorical_crossentropy', metrics=[tf.keras.metrics.Recall()],
                              h3_status = 'deactive', units_h3 = 50, epochs = 100, batch_size = 100)

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

hyp_true = Object.binary_to_single_column_label(y_test)  
Object.plot_comparative_hyp(hyp_true = hyp_true, hyp_pred = y_pred, mark_REM = 'active')

########==================== Plot subjectve hypnos ====================########

Object.plot_subjective_hypno(y_true=y_test, y_pred=y_pred, 
                             test_subjects_list=test_subjects_list,
                             subjects_data_dic=subjects_dic,
                             save_fig = True, 
                             directory="P:/3013080.01/Autoscoring/ssccoorriinngg/")

########=================== Plot overall conf-mat =======================######

Object.plot_confusion_matrix(y_test,y_pred, target_names = ['Wake','N1','N2','SWS','REM'],
                          title='Confusion matrix of ssccoorriinngg algorithm',
                          cmap = None,
                          normalize=True)

########================== Plot subjective conf-mat  ==================########

Object.plot_confusion_mat_subjective(y_true=y_test, y_pred=y_pred, 
                             test_subjects_list=test_subjects_list,
                             subjects_data_dic=subjects_dic)

########========================== Save figure =======================#########
Object.save_figure(saving_format = '.png',
                   directory="P:/3013080.02/Mahdad/Github/ssccoorriinngg/",
                   saving_name = 'test_subject_all' + str(c_subj), dpi = 900,
                   full_screen = False)



