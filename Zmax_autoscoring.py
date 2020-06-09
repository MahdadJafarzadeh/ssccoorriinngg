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
#####==================== Defining required paths r=======================#####

Main_path         = "P:/3013080.01/"
subject_Id_folder = Main_path + "Autoscoring/ssccoorriinngg/"
Data_folder       = Main_path + "Zmax_Data/"
Hypnogram_folder  = Main_path + "zmax_scorings/consensus/"

#####===================== Reading EDF data files=========================#####

subject_ids = loadtxt(subject_Id_folder+"Zmax/subjects_id.txt", dtype = 'str',delimiter='\n')

#####============= create an object of ssccoorriinngg class ==============#####

Object = ssccoorriinngg(filename='', channel='', fs = 256, T = 30)

 #Initialization
subjects_dic     = {}
hyp_dic          = {}
metrics_per_fold = {}
raw_data_dic     = {}
tic_tot = time.time()

# Igonre unnecessary warnings
np.seterr(divide='ignore', invalid='ignore')

#####============= Iterate through each subject to find data =============#####

for idx, c_subj in enumerate(subject_ids):

    # Report the investigational file  
    print (f'Analyzing Subject Number: {c_subj}')
    
    # Separate subject ID and corresponding night
    subj_night_folder = c_subj[0] + "_" + c_subj[1:3] + "/" + c_subj[4:10] + "/"
    
    ## Read in data
    tic      = time.time()
    file     = Data_folder + subj_night_folder
    
    # Reading EEG left and right
    data_L     = mne.io.read_raw_edf(file + "EEG L.edf")
    data_R     = mne.io.read_raw_edf(file + "EEG R.edf")
    
    # Data raw EEG --> Deactive
    # data_L.plot(duration = 30, highpass = .3 , lowpass = 25 )
    
    # Get EEG data from left and right channels
    raw_data_L = data_L.get_data()
    raw_data_R = data_R.get_data()
    
    # Combine channels
    raw_data = np.concatenate((raw_data_L, raw_data_R))
    
    print('Time to read EDF: {}'.format(time.time()-tic))
    
#####=================== Reading acceleration data =======================#####
    # Read Acc per axis and also norm of Acc
    AccNorm, Acc = Object.Read_Acceleration_data(folder_acc = Data_folder + subj_night_folder,
                                                 axis_files = ["dX", "dY", "dZ"],
                                                 file_format = ".edf", plot_Acc = False)
    
    
#####=================Retrieving information from data====================#####
    
# =============================================================================
#     DataInfo          = raw_data_L.info
#     AvailableChannels = DataInfo['ch_names']
#     fs                = int(DataInfo['sfreq'])
# =============================================================================
    
#####===== Sampling rate is 256Hz; thus 1 epoch(30s) is 6000 samples =====#####
    
    fs = 256 #Hz       
    T  = 30 #secs
    len_epoch   = fs * T
    n_channels  = 2  
#####============ Cut tail; use modulo to find full epochs ===============#####
    
    raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
    
#####========== Reshape data [n_channel, len_epoch, n_epochs] ============#####
    
    data_epoched = np.reshape(raw_data,
                             (n_channels, len_epoch,
                             int(raw_data.shape[1]/len_epoch)), order='F' )
    
#####===================== Reading hypnogram data ========================#####
    
    hyp = loadtxt(subject_Id_folder+"Zmax/"+c_subj+".txt", dtype = 'str',delimiter='\t')
    
    Object.find_unscored(hyp, subject_no =c_subj)
    
    #%% Analysis section
#####================= remove chanbnels without scroing ==================#####   
    
    # assign the proper data and labels
    x_tmp_init = data_epoched
    y_tmp_init = hyp
    
    # Ensure equalituy of length for arrays:
    Object.Ensure_data_label_length(x_tmp_init, y_tmp_init)
    
    # Remove non-scored epochs
    x_tmp, y_tmp =  Object.remove_channels_without_scoring(hypno_labels = y_tmp_init,
                                              input_feats = x_tmp_init)
    
    # Remove disconnections
    x_tmp, y_tmp =  Object.remove_disconnection(hypno_labels= y_tmp, 
                                                input_feats=x_tmp) 
    
#####============= Create a one hot encoding form of labels ==============##### 

    # Create binary labels array
    yy = Object.One_hot_encoding(y_tmp)     
    
    # Ensure all the input labels have a class
    Object.Unlabaled_rows_detector(yy)
    
    # Initialize feature array:
    Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
      
#####================== Extract the relevant features ====================#####    
    print(f'Extracting features of subject {c_subj} ....')
    
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
    
    # Defining dictionary to save EEG raw data PER SUBJECT
    raw_data_dic["subject{}".format(c_subj)] = x_tmp
    
    
#####=============== Removing variables for next iteration ===============#####      
    del x_tmp, y_tmp, feat_temp, yy
    toc = time.time()
    
    print('Feature extraction of subject {c_subj} has been finished.')   

print('Total feature extraction of subjects took {tic_tot - time.time()} secs.')
#%% Save created features and labels

#####====================== Save extracted features ======================#####      

path     = project_folder +"3013080.02/ml_project/features/"

filename = 'sleep_scoring_Fp1-Fp2_030620_IncludeContaminatedStagesWithArtefact_ExcludeBadsignal&Unscored'
Object.save_dictionary(path, filename, hyp_dic, subjects_dic)

#####====================== Save raw data as pickle ======================#####      

path     = project_folder + "features/"

filename = 'sleep_scoring_Fp1-Fp2_030620_IncludeContaminatedStagesWithArtefact_ExcludeBadsignal&Unscored_RawData'
Object.save_dictionary(path, filename, hyp_dic, raw_data_dic)


