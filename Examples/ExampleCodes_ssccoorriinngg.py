#%% Import libs
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import h5py
import time
from ssccoorriinngg import ssccoorriinngg
import numpy as np
from sklearn.model_selection import cross_validate


#%% Picking featureset of interest and apply classification
Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)
path   = 'C:/PhD/ML in depression/'
fname  = 'feat42_Fp1-Fp2_train'
feats  = 'featureset'
labels = 'labels'
# Train set
X_train, y_train   = Object.LoadFeatureSet(path, fname, feats, labels)
# Test set
fname  = 'feat42_Fp1-Fp2_test'
X_test, y_test   = Object.LoadFeatureSet(path, fname, feats, labels)

# Define the scoring criteria:
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}   
# Cross-validation using logistic Random Forests
y_pred_RF  = Object.RandomForest_Modelling(X_train, y_train, X_test, y_test, scoring = scoring, n_estimators = 500, cv = 10)
Acc, Recall, prec, f1_sc = Object.multi_label_confusion_matrix(y_test, y_pred_RF)
# Cross-validation using XGBoost
y_pred_xgb  = Object.XGB_Modelling(X_train, y_train,X_test, y_test, scoring, n_estimators = 1000, 
                      cv = 10 , max_depth=3, learning_rate=.1)
Acc, Recall, prec, f1_sc = Object.multi_label_confusion_matrix(y_test, y_pred_xgb)
#%% Outcome measures
# Defien required metrics here:
Metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']
for metric in Metrics:
    #RF
    r1      = results_RF[metric].mean()
    std1    = results_RF[metric].std()
    print(f'{metric} for RF is: {round(r1*100, 2)}+- {round(std1*100, 2)}')
    # xgb
    r2      = results_xgb[metric].mean()
    std2    = results_xgb[metric].std()
    print(f'{metric} for xgb is: {round(r2*100, 2)}+- {round(std2*100, 2)}')
    # SVM
    r3      = results_SVM[metric].mean()
    std3    = results_SVM[metric].std()
    print(f'{metric} for SVM is: {round(r3*100, 2)}+- {round(std3*100, 2)}')
    # LR
    r4      = results_LR[metric].mean()
    std4    = results_LR[metric].std()
    print(f'{metric} for LR is: {round(r4*100, 2)}+- {round(std4*100, 2)}')
#%% Applying Randomized grid search to find the best config. of RF

BestParams_RandomSearch, Bestsocre_RandomSearch ,means, stds, params= Object.RandomSearchRF(X, y,
                        estimator = RandomForestClassifier(), scoring = scoring,
                        n_estimators = [int(x) for x in np.arange(10, 500, 20)],
                        max_features = ['log2', 'sqrt'],
                        max_depth = [int(x) for x in np.arange(10, 100, 30)],
                        min_samples_split = [2, 5, 10],
                        min_samples_leaf = [1, 2, 4],
                        bootstrap = [True, False],
                        n_iter = 100, cv = 10)

#%% Test feature selection methods ##
# PCA
PCA_out                            = Object.FeatSelect_PCA(X, y, n_components = 5)
# Boruta
ranks_Boruta, Feat_selected_Boruta = Object.FeatSelect_Boruta(X, y, max_depth = 7)
# Lasso
Feat_selected_lasso                = Object.FeatSelect_LASSO(X, y, C = 1)
#ANOVA
Feat_selected_ANOVA                = Object.FeatSelect_ANOVA(X,y, k = 80)
#Recruisive
ranks_rec, Feat_selected_rec       = Object.FeatSelect_Recrusive(X, y, k = 20)
#### NOW TEST CLASSIFIERS WITH SELECTED FEATS
results_RF  = Object.RandomForest_Modelling(Feat_selected_Boruta, y, scoring = scoring, n_estimators = 200, cv = 10)


#%% Example save featureset
path = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
Object.SaveFeatureSet(X, y, path = path, filename = 'feat42_N3')

#%% Example load features:
X, y= Object.LoadFeatureSet(path = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/',
                            fname = 'feat42_N3_fp2-M1', 
                            feats = 'featureset', 
                            labels = 'labels')

#%% Combining some REM and SWS epochs

Object.CombineEpochs(directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/',
              ch = 'fp1-M2', N3_fname  = 'tr90_N3_fp1-M2_fp2-M1',
              REM_fname = 'tr90_fp1-M2_fp2-M1',
              saving = True, fname_save = 'tr90_N3&REM_fp1-M2')

#%% How to save some results?
directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/results/' 
fname = '42feats_N3'
with h5py.File((directory+fname + '.h5'), 'w') as wf:
                # Accuracies
                dset = wf.create_dataset('acc_SVM', results_SVM['test_accuracy'].shape, data = results_SVM['test_accuracy'])
                dset = wf.create_dataset('acc_LR' , results_LR['test_accuracy'].shape, data  = results_LR['test_accuracy'])
                dset = wf.create_dataset('acc_RF' , results_RF['test_accuracy'].shape, data  = results_RF['test_accuracy'])
                dset = wf.create_dataset('acc_xgb', results_xgb['test_accuracy'].shape, data = results_xgb['test_accuracy'])
                # Precision
                dset = wf.create_dataset('prec_SVM', results_SVM['test_precision'].shape, data = results_SVM['test_precision'])
                dset = wf.create_dataset('prec_LR' , results_LR['test_precision'].shape, data  = results_LR['test_precision'])
                dset = wf.create_dataset('prec_RF' , results_RF['test_precision'].shape, data  = results_RF['test_precision'])
                dset = wf.create_dataset('prec_xgb', results_xgb['test_precision'].shape, data = results_xgb['test_precision'])
                # Recall
                dset = wf.create_dataset('rec_SVM', results_SVM['test_recall'].shape, data = results_SVM['test_recall'])
                dset = wf.create_dataset('rec_LR' , results_LR['test_recall'].shape, data  = results_LR['test_recall'])
                dset = wf.create_dataset('rec_RF' , results_RF['test_recall'].shape, data  = results_RF['test_recall'])
                dset = wf.create_dataset('rec_xgb', results_xgb['test_recall'].shape, data = results_xgb['test_recall'])
                # f1-score
                dset = wf.create_dataset('f1_SVM', results_SVM['test_f1_score'].shape, data = results_SVM['test_f1_score'])
                dset = wf.create_dataset('f1_LR' , results_LR['test_f1_score'].shape, data  = results_LR['test_f1_score'])
                dset = wf.create_dataset('f1_RF' , results_RF['test_f1_score'].shape, data  = results_RF['test_f1_score'])
                dset = wf.create_dataset('f1_xgb', results_xgb['test_f1_score'].shape, data = results_xgb['test_f1_score'])

#%% Extracting features from more than one channel:
tic = time.time()                
                ########### Central electrodes #############
main_path = "D:/1D_TimeSeries/raw_EEG/without artefact/train_test/"
save_path = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'

fname_C_N3  = (main_path+"tr90_N3_C3-M2_C4-M1.h5")
fname_C_REM = (main_path+"tr90_REM_C3-M2_C4-M1.h5")
ch_C4 = 'C4-M1'
ch_C3 = 'C3-M2'

Object_C3_REM = ML_Depression(filename=fname_C_REM, channel = ch_C3, fs = 200, T = 30)
X_C3_REM,y_C3_REM            = Object_C3_REM.FeatureExtraction() 
Object_C3_REM.SaveFeatureSet(X = X_C3_REM, y=y_C3_REM, path = save_path, filename = 'feat42_C3_REM')
            
Object_C4_REM = ML_Depression(filename=fname_C_REM, channel = ch_C4, fs = 200, T = 30)
X_C4_REM,y_C4_REM            = Object_C4_REM.FeatureExtraction()        
Object_C4_REM.SaveFeatureSet(X = X_C4_REM, y=y_C4_REM, path = save_path, filename = 'feat42_C4_REM')

Object_C3_N3  = ML_Depression(filename=fname_C_N3, channel = ch_C3, fs = 200, T = 30)
X_C3_N3,y_C3_N3            = Object_C3_N3.FeatureExtraction()      
Object_C3_N3.SaveFeatureSet(X = X_C3_N3, y=y_C3_N3, path = save_path, filename = 'feat42_C3_N3')

Object_C4_N3 = ML_Depression(filename=fname_C_N3, channel = ch_C4, fs = 200, T = 30)
X_C4_N3,y_C4_N3            = Object_C4_N3.FeatureExtraction()     
Object_C4_N3.SaveFeatureSet(X = X_C4_N3, y=y_C4_N3, path = save_path, filename = 'feat42_C4_N3')


                ########### Occipital electrodes #############
main_path = "D:/1D_TimeSeries/raw_EEG/without artefact/train_test/"
fname_O_N3  = (main_path+"tr90_N3_O1-M2_O2-M1.h5")
fname_O_REM = (main_path+"tr90_REM_O1-M2_O2-M1.h5")
ch_O2 = 'O2-M1'
ch_O1 = 'O1-M2'
Object_O1_REM = ML_Depression(filename=fname_O_REM, channel = ch_O1, fs = 200, T = 30)
X_O1_REM,y_O1_REM            = Object_O1_REM.FeatureExtraction() 
Object_O1_REM.SaveFeatureSet(X = X_O1_REM, y=y_O1_REM, path = save_path, filename = 'feat42_O1_REM')
              
Object_O2_REM = ML_Depression(filename=fname_O_REM, channel = ch_O2, fs = 200, T = 30)
X_O2_REM,y_O2_REM            = Object_O2_REM.FeatureExtraction()        
Object_O2_REM.SaveFeatureSet(X = X_O2_REM, y=y_O2_REM, path = save_path, filename = 'feat42_O2_REM')

Object_O1_N3  = ML_Depression(filename=fname_O_N3, channel = ch_O1, fs = 200, T = 30)
X_O1_N3,y_O1_N3            = Object_O1_N3.FeatureExtraction()      
Object_O1_N3.SaveFeatureSet(X = X_O1_N3, y=y_O1_N3, path = save_path, filename = 'feat42_O1_N3')

Object_O2_N3 = ML_Depression(filename=fname_O_N3, channel = ch_O2, fs = 200, T = 30)
X_O2_N3,y_O2_N3            = Object_O2_N3.FeatureExtraction()       
Object_O2_N3.SaveFeatureSet(X = X_O2_N3, y=y_O2_N3, path = save_path, filename = 'feat42_O2_N3')

                ########### Fp electrodes #############
main_path = "D:/1D_TimeSeries/raw_EEG/without artefact/train_test/"
fname_fp_N3  = (main_path+"tr90_N3_fp1-M2_fp2-M1.h5")
fname_fp_REM = (main_path+"tr90_REM_fp1-M2_fp2-M1.h5")
ch_fp2 = 'fp2-M1'
ch_fp1 = 'fp1-M2'
Object_fp1_REM = ML_Depression(filename=fname_fp_REM, channel = ch_fp1, fs = 200, T = 30)
X_fp1_REM,y_fp1_REM            = Object_fp1_REM.FeatureExtraction() 
Object_fp1_REM.SaveFeatureSet(X = X_fp1_REM, y=y_fp1_REM, path = save_path, filename = 'feat42_fp1_REM')
              
Object_fp2_REM = ML_Depression(filename=fname_fp_REM, channel = ch_fp2, fs = 200, T = 30)
X_fp2_REM,y_fp2_REM            = Object_fp2_REM.FeatureExtraction()        
Object_fp2_REM.SaveFeatureSet(X = X_fp2_REM, y=y_fp2_REM, path = save_path, filename = 'feat42_fp2_REM')

Object_fp1_N3  = ML_Depression(filename=fname_fp_N3, channel = ch_fp1, fs = 200, T = 30)
X_fp1_N3,y_fp1_N3            = Object_fp1_N3.FeatureExtraction()      
Object_fp1_N3.SaveFeatureSet(X = X_fp1_N3, y=y_fp1_N3, path = save_path, filename = 'feat42_fp1_N3')

Object_fp2_N3 = ML_Depression(filename=fname_fp_N3, channel = ch_fp2, fs = 200, T = 30)
X_fp2_N3,y_fp2_N3            = Object_fp2_N3.FeatureExtraction()     
Object_fp2_N3.SaveFeatureSet(X = X_fp2_N3, y=y_fp2_N3, path = save_path, filename = 'feat42_fp2_N3')
toc = time.time()
print(f'time taken: {toc - tic}')
########## Concatenate all features #########
# RIGHT hemisphere - REM
X_rh_REM = np.column_stack((X_fp2_REM,X_C4_REM))
X_rh_REM = np.column_stack((X_rh_REM,X_O2_REM))
# RIGHT hemisphere - N3
X_rh_N3 = np.column_stack((X_fp2_N3,X_C4_N3))
X_rh_N3 = np.column_stack((X_rh_N3,X_O2_N3))
# LEFT hemisphere - REM
X_lh_REM = np.column_stack((X_fp1_REM,X_C3_REM))
X_lh_REM = np.column_stack((X_lh_REM,X_O1_REM))
# LEFT hemisphere - N3
X_lh_N3 = np.column_stack((X_fp1_N3,X_C3_N3))
X_lh_N3 = np.column_stack((X_lh_N3,X_O1_N3))

# Both sides - REM
X_REM = np.column_stack((X_rh_REM, X_lh_REM))
# Both sides - N3
X_N3 = np.column_stack((X_rh_N3, X_lh_N3))
# Combine SWS and REM
X_SWS_REM = np.row_stack((X_N3, X_REM))
y_SWS_REM = np.concatenate((y_fp2_N3, y_fp2_REM))
# SAVE ALL COMBINATIONS
Object = ML_Depression(filename='', channel='', fs = 200, T = 30)
# one hemisphere
Object.SaveFeatureSet(X = X_rh_REM, y=y_fp2_REM, path = save_path, filename = 'feat42_rh_REM')
Object.SaveFeatureSet(X = X_lh_REM, y=y_fp2_REM, path = save_path, filename = 'feat42_lh_REM')
Object.SaveFeatureSet(X = X_rh_N3 , y=y_fp2_N3 , path = save_path, filename = 'feat42_rh_N3')
Object.SaveFeatureSet(X = X_lh_N3 , y=y_fp2_N3 , path = save_path, filename = 'feat42_lh_N3')
# Both hemisphere
Object.SaveFeatureSet(X = X_N3 , y=y_fp2_N3 , path = save_path, filename = 'feat42_l&rh_N3')
Object.SaveFeatureSet(X = X_REM , y=y_fp2_N3 , path = save_path, filename = 'feat42_l&rh_REM')
# Both hemispheres- SWS &REM combination
Object.SaveFeatureSet(X = X_SWS_REM , y=y_SWS_REM , path = save_path, filename = 'feat42_l&rh_N3&REM')

#%% Load features from different brain regions, sleep stage and combine them
Object              = ML_Depression(filename='', channel='', fs = 200, T = 30)
path                = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
save_path = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
feats               = 'featureset'
labels              = 'labels'
# Pick right hemisphere N3
fname_rh_N3         = 'feat42_rh_N3'
X_rh_N3, y_rh_N3    = Object.LoadFeatureSet(path, fname_rh_N3, feats, labels)
# Pick left hemisphere N3
fname_lh_N3         = 'feat42_lh_N3'
X_lh_N3, y_lh_N3    = Object.LoadFeatureSet(path, fname_lh_N3, feats, labels)
# Pick right hemisphere REM
fname_rh_REM         = 'feat42_rh_REM'
X_rh_REM, y_rh_REM   = Object.LoadFeatureSet(path, fname_rh_REM, feats, labels)
# Pick LEFT hemisphere REM
fname_lh_REM         = 'feat42_lh_REM'
X_lh_REM, y_lh_REM   = Object.LoadFeatureSet(path, fname_lh_REM, feats, labels)
# Combine them
X_N3 = np.column_stack((X_rh_N3, X_lh_N3))

X_REM = np.column_stack((X_rh_REM, X_lh_REM))
# Save combination
Object.SaveFeatureSet(X = X_N3 , y=y_lh_N3 , path = save_path, filename = 'feat42_l&rh_N3')
Object.SaveFeatureSet(X = X_REM , y=y_lh_REM , path = save_path, filename = 'feat42_l&rh_REM')
