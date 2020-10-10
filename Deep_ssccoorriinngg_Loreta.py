# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:41:54 2020

CopyRight: Mahdad Jafarzadeh 2020

Automatic sleep scoring implemented by ssccoorriinngg package.

DEEP LEARNING ON **LORETA DATA** ... 

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
#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 256, T = 30)

# Define path
main_path         = "D:/Loreta_data/features/"
main_path         = "/project/3013080.02/Loreta_data/"

# Distinguishing patients from control group 
gp = loadtxt(main_path + "grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control

# Just keep controls 
for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))
        
#%% Load featureset and labels
filename              =  "Loreta_EpochedDataForDeepLearning_Fp1-Fp2_RawData_NoFilter_071020"
subjects_dic, hyp_dic = Object.load_dictionary(main_path, filename)

# train size
train_size = .8
n_train = round(train_size * len(subj_c))

#######=== Randomly shuffle subjects to choose train and test splits ===#######

subj_c = np.random.RandomState(seed=42).permutation(subj_c)

#######=============== Initialize train and test arrays ================#######c
size_raw_data = np.shape(subjects_dic['subject14'])

X_train = np.empty((size_raw_data[0], size_raw_data[1], 0))
X_test  = np.empty((size_raw_data[0], size_raw_data[1], 0))
y_train = np.empty((0, np.shape(hyp_dic['hyp14'])[1]))
y_test  = np.empty((0, np.shape(hyp_dic['hyp14'])[1]))

########======= Picking the train subjetcs and concatenate them =======########
tic = time.time()
train_subjects_list = []
for c_subj in subj_c[0:n_train]:
    
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
    
    # Keep the train subject
    train_subjects_list.append(str_train_feat)
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
    X_test = np.concatenate((X_test, tmp_x), axis = 2)
    y_test = np.row_stack((y_test, tmp_y))
    
    # keep the subject id
    test_subjects_list.append(str_test_feat)
    
    # remove for next iteration
    del tmp_x, tmp_y, str_test_feat, str_test_hyp
    
print('Test set was successfully created in : {} secs'.format(time.time()-tic))

print(f'Raw train and test data were created.')

# Train model and predict 
y_pred, history = Object.CNN_LSTM_stack_calssifier(X_train, X_test, y_train, y_test, fs=200,path =main_path,\
                                          n_filters = [8, 16, 32], 
                                            kernel_size = [50, 8, 8], LSTM_units = 64, n_LSTM_layers = 4,
                                            recurrent_dropout = .5,loss='categorical_crossentropy', 
                                            optimizer='adam',metrics = ['accuracy'],
                                            epochs = 100, batch_size = 32, verbose = 1,
                                            show_summarize =True, plot_model_graph =False, show_shapes = False,\
                                            patience= 6)

# Save predictions
path     = main_path  
filename = "y_pred_test_CNN_LSTM_Stack_Fp1-Fp2_RawData_Batch=32_epoch=100_train=80%-"
Object.save_dictionary(path, filename, y_pred, y_test)

# Prediction results
Acc, Recall, prec, f1_sc, kappa, mcm= Object.multi_label_confusion_matrix(y_test, y_pred)

