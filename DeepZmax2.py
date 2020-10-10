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

Main_path         = "F:/Zmax_Data/"
subject_Id_folder = Main_path + "features/"
Data_folder       = Main_path + "Zmax_Data/"
Hypnogram_folder  = Main_path + "somno_scorings/Rathiga/"

#####===================== Reading EDF data files=========================#####

subject_ids   = loadtxt(subject_Id_folder+"Subject_ids_excluding 22_2.txt", dtype = 'str',delimiter='\n')

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 256, T = 30)

#%% Load featureset and labels

path     = subject_Id_folder
filename              =  "Zmax_Rathiga_scorings_RawData_ch1-ch2+AccFeats_190620"
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
X_train = np.empty((1, np.shape(subjects_dic[sample_subject])[1], 0))
X_test  = np.empty((1, np.shape(subjects_dic[sample_subject])[1], 0))
y_train = np.empty((0, np.shape(hyp_dic[sample_hyp])[1]))
y_test  = np.empty((0, np.shape(hyp_dic[sample_hyp])[1]))

########======= Picking the train subjetcs and concatenate them =======########
tic = time.time()
train_subjects_list =  ["P_12_night1_scoring.csv.spisop.new - Copy",
                        "P_13_night2_scoring.csv.spisop.new - Copy",
                        "P_15_night2_scoring.csv.spisop.new - Copy",
                        "P_16_night1_scoring.csv.spisop.new - Copy",
                        "P_18_night1_scoring.csv.spisop.new - Copy",
                        "P_20_night1_scoring.csv.spisop.new - Copy",
                        "P_21_night1_scoring.csv.spisop.new - Copy",
                        "P_23_night1_scoring.csv.spisop.new - Copy"]

for c_subj in train_subjects_list:
    
    # train hypnogram
    str_train_hyp  = 'hyp' + str(c_subj)
    
    # train featureset
    str_train_feat = 'subject' + str(c_subj)
    
    # create template arrays for featurs and label
    tmp_x          =  subjects_dic[str_train_feat]
    tmp_y          =  hyp_dic[str_train_hyp]
    
    # Concatenate features and labels
    X_train = np.concatenate((X_train, tmp_x), axis = 2)
    y_train = np.row_stack((y_train, tmp_y))
    
    del tmp_x, tmp_y
    
print('Training set was successfully created in : {} secs'.format(time.time()-tic))

 #%% ================================Test part==============================%%#

########======== Picking the test subjetcs and concatenate them =======########
tic                = time.time()
test_subjects_list = []
tst_subj_list = ["P_12_night2_scoring.csv.spisop.new - Copy",
                "P_12_night3_scoring.csv.spisop.new - Copy",
                "P_13_night3_scoring.csv.spisop.new - Copy",
                "P_14_night3_scoring.csv.spisop.new - Copy",
                "P_15_night3_scoring.csv.spisop.new - Copy",
                "P_16_night3_scoring.csv.spisop.new - Copy",
                "P_18_night2_scoring.csv.spisop.new - Copy",
                "P_18_night3_scoring.csv.spisop.new - Copy",
                "P_20_night2_scoring.csv.spisop.new - Copy",
                "P_20_night3_scoring.csv.spisop.new - Copy",
                "P_21_night2_scoring.csv.spisop.new - Copy",
                "P_21_night3_scoring.csv.spisop.new - Copy"]

for c_subj in tst_subj_list:
   
    # test hypnogram
    str_test_hyp  = 'hyp' + str(c_subj)
    
    # test featureset
    str_test_feat = 'subject' + str(c_subj)
    
    # create template arrays for featurs and  label
    tmp_x         =  subjects_dic[str_test_feat]
    tmp_y         =  hyp_dic[str_test_hyp]
    
    # Concatenate features and labels
    X_test = np.concatenate((X_test, tmp_x), axis = 2)
    y_test = np.row_stack((y_test, tmp_y))
    
    # keep the subject id
    test_subjects_list.append(str_test_feat)
    
    # remove for next iteration
    del tmp_x, tmp_y, str_test_feat, str_test_hyp
    
print('Test set was successfully created in : {} secs'.format(time.time()-tic))

print(f'Raw train and test data were created.')

########================== Replace any probable NaN ===================########
# =============================================================================
# 
# X_train = Object.replace_NaN_with_mean(X_train)
# X_test  = Object.replace_NaN_with_mean(X_test)
# 
# ########================== Replace any probable inf ===================########
# 
# X_train = Object.replace_inf_with_mean(X_train)
# X_test  = Object.replace_inf_with_mean(X_test)
# =============================================================================

########==================== Z-score of features ======================########

#X_train, X_test = Object.Standardadize_features(X_train, X_test)

########====================== Feature Selection ======================########

y_train_td = Object.binary_to_single_column_label(y_train)


##### CNN 1 
model = Object.CNN_LSTM_stack_calssifier(X_train, y_train, fs=256, n_filters = [8, 16, 32], 
                        kernel_size = [50, 8, 8], LSTM_units = 64, n_LSTM_layers = 4,
                        recurrent_dropout = .3,loss='mean_squared_error', 
                        optimizer='adam',metrics = ['accuracy'],
                        epochs = 10, batch_size = 128, verbose = 1,
                        show_summarize =True, show_shapes = False)

########============== Define classifier of interest ==================########
premodel = Object.CRNN_premodel_classifier( X_train, y_train, fs=256, n_filters = [8, 16, 32], 
                        kernel_size = [50, 8, 8], loss='mean_squared_error', 
                        optimizer='adam',metrics = [tf.keras.metrics.Recall()],
                        epochs = 50, batch_size = 128, verbose = 1,
                        show_summarize =True, plot_model_graph =True, show_shapes = False)

main_model = Object.CRNN_main_classifier(premodel, X_train, y_train, fs=256, before_flatten_layer=12,
                            loss='mean_squared_error', 
                            LSTM_units = 64, recurrent_dropout = .3,
                            optimizer='adam',metrics = [tf.keras.metrics.Recall()],
                            epochs = 5, batch_size = 256, verbose = 1,
                            show_summarize =True, plot_model_graph =True, show_shapes = False)

main_model.fit(np.transpose(X_train), y_train, epochs=5, batch_size=256, verbose=1)
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



