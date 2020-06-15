# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:07:37 2020

@author: mahjaf

Using this code, one can directly feed in EDF data and select channels of interest
to perform classification --> USING DEEP LEARNING 

Please Note: we recommend to use "EDF_to_h5.py" to firstly convert EDF into
a lighter data format to accelerate computations; however, this code is only
meant for those that want to skip this conversion and directly choose EDF data
as input.
"""
#%% Reading EDF section
#####===================== Importiung libraries =========================#####
import tensorflow as tf
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

#####===================== Defining project folder========================#####

project_folder = "F:/Loreta_data/"

#####===================== Reading EDF data files=========================#####

pat_labels = loadtxt(project_folder+"patient_labels.txt", delimiter="\t", skiprows = 1)

#####============= Distinguishing patients from control group=============#####

gp = loadtxt(project_folder+"grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients

for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))
    elif c[1] == 'CC':
        pass
    else:
        subj_p.append(int(c[0]))

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)

#%% Load Raw data for deep learning

path     = project_folder +"features/"
filename              =  "sleep_scoring_Fp1-Fp2_030620_IncludeContaminatedStagesWithArtefact_ExcludeBadsignal&Unscored_RawData"
raw_data_dic, hyp_dic = Object.load_dictionary(path, filename)

#%% ================================Training part==============================

# Training perentage
train_size = .8
n_train = round(train_size * len(subj_c))

#######=== Randomly shuffle subjects to choose train and test splits ===#######

subj_c = np.random.RandomState(seed=42).permutation(subj_c)

#######=============== Initialize train and test arrays ================#######c
size_raw_data = np.shape(raw_data_dic['subject14'])

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
    tmp_x          =  raw_data_dic[str_train_feat]
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
    tmp_x         =  raw_data_dic[str_test_feat]
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

# premodel CRNN
premodel = Object.CRNN_premodel_classifier(X_train, y_train, fs=200, n_filters = [8, 16, 32], 
                        kernel_size = [50, 8, 8], loss='mean_squared_error', 
                        optimizer='adam',metrics = [tf.keras.metrics.Recall()],
                        epochs = 10, batch_size = 128, verbose = 1,
                        show_summarize =True, plot_model_graph =True, show_shapes = False)

# main model CRNN
model = Object.CRNN_main_classifier(premodel, X_train, y_train, fs=200, before_flatten_layer=12,
                            n_filters = [8, 16, 32], 
                            kernel_size = [50, 8, 8], loss='mean_squared_error', 
                            LSTM_units = 64, recurrent_dropout = .3,
                            optimizer='adam',metrics = [tf.keras.metrics.Recall()],
                            epochs = 10, batch_size = 128, verbose = 1,
                            show_summarize =True, plot_model_graph =True, show_shapes = False)






########===== Metrics to assess the model performance on test data ====########

Acc, Recall, prec, f1_sc, kappa, mcm= Object.multi_label_confusion_matrix(y_test, y_pred)

########================= Creating subjective outputs =================########
# =============================================================================
# 
# Object.create_subjecive_results(y_true=y_test, y_pred=y_pred, 
#                                 test_subjects_list = test_subjects_list,
#                                 subjects_data_dic = raw_data_dic,
#                                 fname_save = "results")
# 
# ########============= find number of epochs per stage =================########
# 
# Object.find_number_of_samples_per_class(y_test, including_artefact = False)
# 
# ########================== Comparative hypnogram ======================########
# 
# Object.plot_comparative_hyp(y_true = y_test, y_pred = y_pred, mark_REM = 'active')
# 
# ########==================== Plot subjectve hypnos ====================########
# 
# Object.plot_subjective_hypno(y_true=y_test, y_pred=y_pred, 
#                              test_subjects_list=test_subjects_list,
#                              subjects_data_dic=raw_data_dic,
#                              save_fig = True, 
#                              directory="C:/PhD/Github/ssccoorriinngg/")
# 
# ########=================== Plot overall conf-mat =======================######
# 
# Object.plot_confusion_matrix(y_test,y_pred, target_names = ['Wake','N1','N2','SWS','REM'],
#                           title='Confusion matrix of ssccoorriinngg algorithm',
#                           cmap = None,
#                           normalize=True)
# 
# ########================== Plot subjective conf-mat  ==================########
# 
# Object.plot_confusion_mat_subjective(y_true=y_test, y_pred=y_pred, 
#                              test_subjects_list=test_subjects_list,
#                              subjects_data_dic=raw_data_dic)
# 
# ########========================== Save figure =======================#########
# Object.save_figure(saving_format = '.png',
#                    directory="P:/3013080.02/Mahdad/Github/ssccoorriinngg/",
#                    saving_name = 'test_subject_all' + str(c_subj), dpi = 900,
#                    full_screen = False)
# 
# =============================================================================


