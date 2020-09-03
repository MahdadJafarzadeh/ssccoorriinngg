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
from scipy import signal
#####==================== Defining required paths r=======================#####

Main_path         = "P:/3013080.01/"
subject_Id_folder = Main_path + "Autoscoring/ssccoorriinngg/"
Data_folder       = Main_path + "Zmax_Data/"
Hypnogram_folder  = Main_path + "somno_scorings/Rathiga/"

#####===================== Reading EDF data files=========================#####

subject_ids = loadtxt(subject_Id_folder+"Zmax/Subject_ids_excluding 22_2.txt", dtype = 'str',delimiter='\n')

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 256, T = 30)

#%% Load featureset and labels

path     = "P:/3013080.01/Autoscoring/features/"
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
# path     = "P:/3013080.01/Autoscoring/features/"
# filename              = "Selected_Features_BoturaNoTimeDependency_5_Backward_Zmax_ch1-ch2+Acc_200620"
# with open(path+filename+'.pickle',"wb") as f:
#     pickle.dump(selected_feats_ind, f)
# =============================================================================
     
########################### Load selected feats ###############################

path     = "P:/3013080.01/Autoscoring/features/"
filename              = "Selected_Features_BoturaAfterTD=5_Backward_Zmax_ch1-ch2+Acc_200620"
#filename              = "sleep_scoring_NoArousal_8channels_selected_feats_NEW"
with open(path + filename + '.pickle', "rb") as f: 
    selected_feats_ind = pickle.load(f)
    
########=================== Apply selected features ===================########

X_train = X_train_td[:, selected_feats_ind]
X_test  = X_test_td[:, selected_feats_ind]

########============== Define classifier of interest ==================########
y_pred = Object.XGB_Modelling(X_train, y_train,X_test, y_test, n_estimators = 500)
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
                             save_fig = False, 
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



