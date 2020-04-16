# -*- coding: utf-8 -*-
"""
Created on 29/03/2020 
@author: Mahdad

THIS IS THE CLASS FOR "MACHINE LEARNING & DEPRESSION PROJECT."
ML_Deperession class is made to classify Normal EEG epochs from the ones from
depressed people. Using this code, one can use channels of interest and sleep
stges of interest to perform classification.
The class is capable of extracting relevant features, applying various machine-
learning algorithms and finally applying Randomized grid search to tune hyper-
parameters of different classifiers.

To see the example codes and instructions how to use each method of class, 
please visit: https://github.com/MahdadJafarzadeh/ssccoorriinngg/

to import the class, use:
from ssccoorriinngg import ssccoorriinngg

"""
#%% Importing libs
import numpy as np
import pandas as pd 
import pywt
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch
from sklearn.ensemble import RandomForestClassifier
import heapq
from scipy.signal import argrelextrema
from sklearn.model_selection import cross_val_score,KFold, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from entropy.entropy import spectral_entropy
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from scipy.fftpack import fft
import h5py
import time
import pyeeg
import nolds

class ssccoorriinngg():
    
    def __init__(self, filename, channel, fs, T):
        
        self.filename = filename
        self.channel  = channel
        self.fs       = fs
        self.T        = T
        
    #%% Loading existing featureset
    def LoadFeatureSet(self, path, fname, feats, labels):
        # Reading N3 epochs
        with h5py.File(path + fname + '.h5', 'r') as rf:
            X  = rf['.'][feats].value
            y  = rf['.'][labels].value
        return X, y
    
    #%% Combining epochs
    def CombineEpochs(self, directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/',
                      ch = 'fp2-M1', N3_fname  = 'tr90_N3_fp1-M2_fp2-M1',
                      REM_fname = 'tr90_fp1-M2_fp2-M1',
                      saving = False, fname_save = 'tst'):
        # Initialization 
        tic       = time.time() 
        # Defining the directory of saved files
        directory = directory 
        # Define channel of interest (currently fp1-M2 and fp2-M1 are only active)
        ch = ch
        # N3 epochs Filename 
        N3_fname  = N3_fname
        # REM epochs Filename
        REM_fname = REM_fname
        
        # Reading N3 epochs
        with h5py.File(directory + N3_fname + '.h5', 'r') as rf:
            xtest_N3  = rf['.']['x_test_' + ch].value
            xtrain_N3 = rf['.']['x_train_' + ch].value
            ytest_N3  = rf['.']['y_test_' + ch].value
            ytrain_N3 = rf['.']['y_train_' + ch].value
        print(f'N3 epochs were loaded successfully in {time.time()-tic} secs')    
        
        # Reading REM epochs
        tic       = time.time() 
        with h5py.File(directory + REM_fname + '.h5', 'r') as rf:
            xtest_REM  = rf['.']['x_test_' + ch].value
            xtrain_REM = rf['.']['x_train_' + ch].value
            ytest_REM  = rf['.']['y_test_' + ch].value
            ytrain_REM = rf['.']['y_train_' + ch].value
        print(f'REM epochs were loaded successfully in {time.time()-tic} secs') 
           
        # Combining epochs
        xtest   = np.row_stack((xtest_N3, xtest_REM))
        xtrain  = np.row_stack((xtrain_N3, xtrain_REM))
        ytest   = np.row_stack((ytest_N3, ytest_REM))
        ytrain  = np.row_stack((ytrain_N3, ytrain_REM))
        print('Epochs were successfully concatenated')
        
        # Save concatenated results:
        # SAVE train/test splits
        if saving == True:
            tic = time.time()
            fname_save = fname_save
            with h5py.File((directory+fname_save + '.h5'), 'w') as wf:
                dset = wf.create_dataset('y_test_' +ch, ytest.shape, data=ytest)
                dset = wf.create_dataset('y_train_'+ch, ytrain.shape, data=ytrain)
                dset = wf.create_dataset('x_test_' +ch, xtest.shape, data=xtest)
                dset = wf.create_dataset('x_train_'+ch, xtrain.shape, data=xtrain)
            print('Time to save H5: {}'.format(time.time()-tic))
            return xtrain, ytrain, xtest, ytest
        else:
            print('Outputs were generated but not saved')
            return xtrain, ytrain, xtest, ytest

    #%% Feature extarction
    def FeatureExtraction(self):
        
        ''' ~~~~~~################## INSTRUCTION #################~~~~~~~~
        ----
        THIS IS A FUNCTION TO EXTRACT FEATURES AND THEN USE THEM FOR ANY KIND OF
        SUPERVISED MACHINE LEARNING ALGORITHM.
    
        INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        
        OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".
        '''
        # Loading data section
        # Load data
        tic = time.time() 
        fname = self.filename
        
        # choose channel to extract features from
        ch = self.channel
        fs = self.fs #Hz
        T  = self.T #sec
        # Split train and test 
        with h5py.File(fname, 'r') as rf:
            xtest  = rf['.']['x_test_' + ch].value
            xtrain = rf['.']['x_train_' + ch].value
            ytest  = rf['.']['y_test_' + ch].value
            ytrain = rf['.']['y_train_' + ch].value
        print('train and test data loaded in : {} secs'.format(time.time()-tic))
        
        # Flatten data for filter and normalization
        X_train = np.reshape(xtrain, (np.shape(xtrain)[0] * np.shape(xtrain)[1] ,1))
        X_test  = np.reshape(xtest, (np.shape(xtest)[0] * np.shape(xtest)[1] ,1))
        
        #%% Filtering section
        ## Defining preprocessing function ##
        def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
            nyq = 0.5 * fs
            low = lowcut /nyq
            high = highcut/nyq
            b, a = butter(order, [low, high], btype='band')
            #print(b,a)
            y = lfilter(b, a, data)
            return y
        
        # Apply filter
        X_train = butter_bandpass_filter(data=X_train, lowcut=.1, highcut=30, fs=fs, order=2)
        X_test  = butter_bandpass_filter(data=X_test , lowcut=.1, highcut=30, fs=fs, order=2)
        
        #%% Normalization section - DEACTIVATED
        #sc = StandardScaler()
        #X_train = sc.fit_transform(X_train)
        #X_test  = sc.transform(X_test)
        
        #%% Reshaping data per epoch
        X_train = np.reshape(X_train, (int(len(X_train) / (fs*T)), fs*T))
        X_test  = np.reshape(X_test,  (int(len(X_test) / (fs*T)), fs*T))
        
        # Concatenate to extract feats
        X       = np.concatenate((X_train, X_test))
        
        
        
        #%% Feature Extraction section
        
        # Defining EEG bands:
        eeg_bands = {'Delta'     : (0.5, 4),
                     'Theta_low' : (4  , 6),
                     'Theta_high': (6  , 8),
                     'Alpha'     : (8  , 11),
                     'Beta'      : (16 , 24),
                     'Sigma'     : (12 , 15),
                     'Sigma_slow': (10 , 12)}
        
        # Initializing variables of interest
        eeg_band_fft      = dict()
        freq_ix           = dict()
        Features = np.empty((0, 42))
        # Settings of peridogram    
        Window = 'hann'
        # zero-padding added with respect to (Nfft=2^(nextpow2(len(window))))
        Nfft = 2 ** 15 
        # Defining freq. resoultion
        fm, _ = periodogram(x = X[0,:], fs = fs, nfft = Nfft , window = Window)  
        tic = time.time()
        # Finding the index of different freq bands with respect to "fm" #
        for band in eeg_bands:
            freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                               (fm <= eeg_bands[band][1]))[0]    
        
        
        # Defining for loop to extract features per epoch
        for i in np.arange(len(X)):
            
            data = X[i,:]
            
            # Compute the "total" power inside the investigational window
            _ , pxx = periodogram(x = data, fs = fs, nfft = Nfft , window = Window) 
            
            # Initialization for wavelet 
            cA_values  = []
            cD_values  = []
            cA_mean    = []
            cA_std     = []
            cA_Energy  = []
            cD_mean    = []
            cD_std     = []
            cD_Energy  = []
            Entropy_D  = []
            Entropy_A  = []
            first_diff = np.zeros(len(data)-1)
            
            '''Power in differnt freq ranges ''' 
            # Total pow is defined form 0.5 - 20 Hz
            pow_total      = np.sum(pxx[np.arange(freq_ix['Delta'][0], freq_ix['Beta'][-1]+1)])
            Pow_Delta      = np.sum(pxx[freq_ix['Delta']]) / pow_total
            Pow_Theta_low  = np.sum(pxx[freq_ix['Theta_low']]) / pow_total
            Pow_Theta_high = np.sum(pxx[freq_ix['Theta_high']]) / pow_total
            Pow_Alpha      = np.sum(pxx[freq_ix['Alpha']]) / pow_total
            Pow_Beta       = np.sum(pxx[freq_ix['Beta']])  / pow_total
            Pow_Sigma      = np.sum(pxx[freq_ix['Sigma']]) / pow_total
            Pow_Sigma_slow = np.sum(pxx[freq_ix['Sigma_slow']]) / pow_total
            
            '''Apply Welch to see the dominant Max power in each freq band''' 
            ff, Psd             = welch(x = data, fs = fs, window = 'hann', nperseg= 512, nfft = Nfft)
            Pow_max_Total       = np.max(Psd[np.arange(freq_ix['Delta'][0], freq_ix['Beta'][-1]+1)])
            Pow_max_Delta       = np.max(Psd[freq_ix['Delta']])
            Pow_max_Theta_low   = np.max(Psd[freq_ix['Theta_low']])
            Pow_max_Theta_high  = np.max(Psd[freq_ix['Theta_high']])
            Pow_max_Alpha       = np.max(Psd[freq_ix['Alpha']])
            Pow_max_Beta        = np.max(Psd[freq_ix['Beta']])
            Pow_max_Sigma       = np.max(Psd[freq_ix['Sigma']])
            Pow_max_Sigma_slow  = np.max(Psd[freq_ix['Sigma_slow']])
            
            ''' Spectral Entropy '''
            Entropy_Welch = spectral_entropy(x = data, sf=fs, method='welch', nperseg = 512)
            Entropy_fft   = spectral_entropy(x = data, sf=fs, method='fft')
               
            ''' Wavelet Decomposition ''' 
            cA,cD=pywt.dwt(data,'coif1')
            cA_values.append(cA)
            cD_values.append(cD)
            cA_mean.append(np.mean(cA_values))
            cA_std.append(np.std(cA_values))
            cA_Energy.append(np.sum(np.square(cA_values)))
            cD_mean.append(np.mean(cD_values))
            cD_std.append(np.std(cD_values))
            cD_Energy.append(np.sum(np.square(cD_values)))
            Entropy_D.append(np.sum(np.square(cD_values) * np.log(np.square(cD_values))))
            Entropy_A.append(np.sum(np.square(cA_values) * np.log(np.square(cA_values))))
            
            ''' Hjorth Parameters '''
            hjorth_activity     = np.var(data)
            diff_input          = np.diff(data)
            diff_diffinput      = np.diff(diff_input)
            hjorth_mobility     = np.sqrt(np.var(diff_input)/hjorth_activity)
            hjorth_diffmobility = np.sqrt(np.var(diff_diffinput)/np.var(diff_input))
            hjorth_complexity   = hjorth_diffmobility / hjorth_mobility
             
            ''' Statisctical features'''
            Kurt     = kurtosis(data, fisher = False)
            Skewness = skew(data)
            Mean     = np.mean(data)
            Median   = np.median(data)
            Std      = np.std(data)
            ''' Coefficient of variation '''
            coeff_var = Std / Mean
            
            ''' First and second difference mean and max '''
            sum1  = 0.0
            sum2  = 0.0
            Max1  = 0.0
            Max2  = 0.0
            for j in range(len(data)-1):
                    sum1     += abs(data[j+1]-data[j])
                    first_diff[j] = abs(data[j+1]-data[j])
                    
                    if first_diff[j] > Max1: 
                        Max1 = first_diff[j] # fi
                        
            for j in range(len(data)-2):
                    sum2 += abs(first_diff[j+1]-first_diff[j])
                    if abs(first_diff[j+1]-first_diff[j]) > Max2 :
                    	Max2 = first_diff[j+1]-first_diff[j] 
                        
            diff_mean1 = sum1 / (len(data)-1)
            diff_mean2 = sum2 / (len(data)-2) 
            diff_max1  = Max1
            diff_max2  = Max2
            
            ''' Variance and Mean of Vertex to Vertex Slope '''
            t_max   = argrelextrema(data, np.greater)[0]
            amp_max = data[t_max]
            t_min   = argrelextrema(data, np.less)[0]
            amp_min = data[t_min]
            tt      = np.concatenate((t_max,t_min),axis=0)
            if len(tt)>0:
                tt.sort() #sort on the basis of time
                h=0
                amp = np.zeros(len(tt))
                res = np.zeros(len(tt)-1)
                
                for l in range(len(tt)):
                        amp[l] = data[tt[l]]
                        
                out = np.zeros(len(amp)-1)     
                 
                for j in range(len(amp)-1):
                    out[j] = amp[j+1]-amp[j]
                amp_diff = out
                
                out = np.zeros(len(tt)-1)  
                
                for j in range(len(tt)-1):
                    out[j] = tt[j+1]-tt[j]
                tt_diff = out
                
                for q in range(len(amp_diff)):
                        res[q] = amp_diff[q]/tt_diff[q] #calculating slope        
                
                slope_mean = np.mean(res) 
                slope_var  = np.var(res)   
            else:
                slope_var, slope_mean = 0, 0
                
            ''' Spectral mean '''
            Spectral_mean = 1 / (freq_ix['Beta'][-1] - freq_ix['Delta'][0]) * (Pow_Delta + 
                    Pow_Theta_low + Pow_Theta_high + Pow_Alpha + Pow_Beta + 
                    Pow_Sigma) 
            
            ''' Correlation Dimension Feature '''
            #cdf = nolds.corr_dim(data,1)
            
            ''' Detrended Fluctuation Analysis ''' 

            ''' Wrapping up featureset '''
            feat = [pow_total, Pow_Delta, Pow_Theta_low, Pow_Theta_high, Pow_Alpha,
                    Pow_Beta, Pow_Sigma, Pow_Sigma_slow, cA_mean[0], cA_std[0],
                    cA_Energy[0], cD_Energy[0],  cD_mean[0], cD_std[0],
                    Entropy_D[0], Entropy_A[0], Entropy_Welch, Entropy_fft,
                    Kurt, Skewness, Mean, Median, Spectral_mean, hjorth_activity,
                    hjorth_mobility, hjorth_complexity, Std, coeff_var,
                    diff_mean1, diff_mean2, diff_max1, diff_max2, slope_mean, 
                    slope_var, Pow_max_Total, Pow_max_Delta, Pow_max_Theta_low,
                    Pow_max_Theta_high, Pow_max_Alpha, Pow_max_Beta, Pow_max_Sigma,
                    Pow_max_Sigma_slow]
            
            Features = np.row_stack((Features,feat))
            
        #%% Replace the NaN values of features with the mean of each feature column
        print('Features were successfully extracted in: {} secs'.format(time.time()-tic))
        
        aa, bb = np.where(np.isnan(Features))
        for j in np.arange(int(len(aa))):
            Features[aa[j],bb[j]] = np.nanmean(Features[:,bb[j]])
        print('the NaN values were successfully replaced with the mean of related feature.')    
        #%% Normalizing features
        Feat_train = Features[:int(len(X_train)),:]
        Feat_test = Features[int(len(X_train)):,:]
        sc = StandardScaler()
        Feat_train = sc.fit_transform(Feat_train)
        Feat_test = sc.transform(Feat_test)
        
        #%% Shuffle train and test data with rand perumtation
        rp_train = np.random.RandomState(seed=42).permutation(len(Feat_train))
        rp_test  = np.random.RandomState(seed=42).permutation(len(Feat_test))
        
        Feat_train_rp = Feat_train[rp_train,:]
        Feat_test_rp  = Feat_test[rp_test,:]
        y_train_rp    = ytrain[rp_train,:]
        y_test_rp     = ytest[rp_test,:]
        
        X_train = Feat_train_rp
        X_test  = Feat_test_rp 
        y_train = y_train_rp
        y_test  = y_test_rp
        
        return X_train, X_test, y_train, y_test
    
    #%% Save features
    
    def SaveFeatureSet(self, X, y, path, filename):
        path     = path  
        filename = filename
        with h5py.File((path+filename + '.h5'), 'w') as wf:
            dset = wf.create_dataset('featureset', X.shape, data = X)
            dset = wf.create_dataset('labels', y.shape, data = y)
        print('Features have been successfully saved!')

        ######################## DEFINING FEATURE SELECTION METHODS ######################
    #%% Feature selection section - 1. Boruta method
    def FeatSelect_Boruta(self, X,y,max_depth = 7):
        #import lib
        tic = time.time()
        from boruta import BorutaPy
        #instantiate an estimator for Boruta. 
        rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=max_depth)
        # Initiate Boruta object
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=0)
        # fir the object
        feat_selector.fit(X=X, y=y)
        # Check selected features
        print(feat_selector.support_)
        # Select the chosen features from our dataframe.
        Feat_selected = X[:, feat_selector.support_]
        print(f'Selected Feature Matrix Shape {Feat_selected.shape}')
        toc = time.time()
        print(f'Feature selection using Boruta took {toc-tic}')
        ranks = feat_selector.ranking_
        
        return ranks, Feat_selected
    
    #%% Feature selection using LASSO as regression penalty
    def FeatSelect_LASSO(self, X, y, C = 1):
        from sklearn.linear_model import LogisticRegression
        tic = time.time()
        from sklearn.linear_model import Lasso
        from sklearn.feature_selection import SelectFromModel
        #create object
        sel_ = SelectFromModel(LogisticRegression(C=C, penalty='l1'))
        sel_.fit(X, y)
        # find the selected feature indices
        selected_ = sel_.get_support()
        # Select releavnt features
        Feat_selected = X[:, selected_]
        toc = time.time()
        print(f'Total time for LASSO feature selection was: {toc-tic}')
        print(f'total of {len(Feat_selected)} was selected out of {np.shape(X)[1]} features')
        return Feat_selected
    
    #%% Feature Selection with Univariate Statistical Tests
    def FeatSelect_ANOVA(self, X, y, k=20):
        tic = time.time()
        from sklearn.feature_selection import SelectKBest, f_classif
        test = SelectKBest(score_func=f_classif, k=k)
        fit = test.fit(X, y)
        # summarize scores
        print(f' scores: {fit.scores_}')
        Feat_selected = fit.transform(X)
        toc = time.time()
        print(f'Total time for ANOVA feature selection was: {toc-tic}')

        return Feat_selected
    #%% # Recursive Feature Elimination
    def FeatSelect_Recrusive(self, X,y, k = 20):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
            # feature extraction
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model, n_features_to_select = k)
        fit = rfe.fit(X, y)
        ranks = fit.ranking_
        selected_ = fit.support_
        Feat_selected = X[:, selected_]
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)
        
        return ranks, Feat_selected

    #%% PCA
    def FeatSelect_PCA(self, X, y, n_components = 5):
        tic = time.time()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        PCA_out = pca.fit(X)
        # summarize components
        print("Explained Variance: %s" % PCA_out.explained_variance_ratio_)
        print(PCA_out.components_)
        toc = time.time()
        print(f'Total time for PCA feature selection was: {toc-tic}')

        return PCA_out
    
    ######################## DEFINING SUPERVISED CLASSIFIERs ######################
    
    #%% Random Forest
    def RandomForest_Modelling(self, X_train, y_train,X_test, y_test, scoring, n_estimators = 500, cv = 10):
        tic = time.time()
        classifier_RF = RandomForestClassifier(n_estimators = n_estimators)
        classifier_RF.fit(X_train, y_train)
        results_RF = classifier_RF.predict(X_test)
        #results_RF = cross_validate(estimator = classifier_RF, X = X, 
        #                         y = y, scoring = scoring, cv = KFold(n_splits = cv))
        #Acc_cv10_RF = accuracies_RF.mean()
        #std_cv10_RF = accuracies_RF.std()
        #print(f'Cross validation finished: Mean Accuracy {Acc_cv10_RF} +- {std_cv10_RF}')
        #print('Cross validation for RF took: {} secs'.format(time.time()-tic))
        return results_RF
    
    #%% Kernel SVM
    def KernelSVM_Modelling(self, X, y, cv, scoring, kernel):
        tic = time.time()
        from sklearn.svm import SVC
        classifier_SVM = SVC(kernel = kernel)
        results_SVM = cross_validate(estimator = classifier_SVM, X = X, 
                                 y = y, scoring = scoring, cv = KFold(n_splits = cv))
        #Acc_cv10_SVM = accuracies_SVM.mean()
        #std_cv10_SVM = accuracies_SVM.std()
        #print(f'Cross validation finished: Mean Accuracy {Acc_cv10_SVM} +- {std_cv10_SVM}')
        print('Cross validation for SVM took: {} secs'.format(time.time()-tic))
        return results_SVM
    
    
    #%% Logistic regression
    def LogisticRegression_Modelling(self, X, y, scoring, cv = 10, max_iter = 500):
        tic = time.time()
        from sklearn.linear_model import LogisticRegression
        classifier_LR = LogisticRegression(max_iter = max_iter)
        results_LR = cross_validate(estimator = classifier_LR, X = X, 
                                 y = y, scoring = scoring, cv = KFold(n_splits = cv))
        #Acc_cv10_LR = accuracies_LR.mean()
        #std_cv10_LR = accuracies_LR.std()
        #print(f'Cross validation finished: Mean Accuracy {Acc_cv10_LR} +- {std_cv10_LR}')
        print('Cross validation for LR took: {} secs'.format(time.time()-tic))
        return results_LR
    #%% XGBoost
    def XGB_Modelling(self, X, y, scoring, n_estimators = 250, 
                      cv = 10 , max_depth=3, learning_rate=.1):
        tic = time.time()
        from xgboost import XGBClassifier
        classifier_xgb = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth,
                                       learning_rate = learning_rate)
        results_xgb = cross_validate(estimator = classifier_xgb, X = X, 
                                 y = y, scoring = scoring, cv = KFold(n_splits = cv))
        #Acc_cv10_xgb = accuracies_xgb.mean()
        #std_cv10_xgb = accuracies_xgb.std()
        #print(f'Cross validation finished: Mean Accuracy {Acc_cv10_xgb} +- {std_cv10_xgb}')
        print('Cross validation for xgb took: {} secs'.format(time.time()-tic))
        return results_xgb
    
    #%% ANN
    def ANN_Modelling(self, X, y, units_h1,  input_dim, units_h2, units_output,
                  init = 'uniform', activation = 'relu', optimizer = 'adam',
                  loss = 'binary_crossentropy', metrics = ['accuracy'],
                  h3_status = 'deactive', units_h3 = 50):
        # Importing the Keras libraries and packages
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        
        # Initialising the ANN
        classifier = Sequential()
        
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = units_h1, init = init, activation = activation, input_dim = input_dim))
        
        # Adding the second hidden layer
        classifier.add(Dense(units = units_h2 , init = init, activation = activation))
        
        # Adding the third hidden layer
        if h3_status == 'active':
            classifier.add(Dense(units = units_h3 , init = init, activation = activation))
            
        # Adding the output layer
        classifier.add(Dense(units = units_output, init = init, activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = loss , metrics = metrics)
        
        return classifier

    #%% Randomized and grid search 
    ######################## DEFINING RANDOMIZED SEARCH ###########################
    #       ~~~~~~!!!!! THIS IS FOR RANDOM FOREST AT THE MOMENT ~~~~~~!!!!!!
    def RandomSearchRF(self, X, y, scoring, estimator = RandomForestClassifier(),
                        n_estimators = [int(x) for x in np.arange(10, 500, 20)],
                        max_features = ['log2', 'sqrt'],
                        max_depth = [int(x) for x in np.arange(10, 100, 30)],
                        min_samples_split = [2, 5, 10],
                        min_samples_leaf = [1, 2, 4],
                        bootstrap = [True, False],
                        n_iter = 100, cv = 10):
        from sklearn.model_selection import RandomizedSearchCV
        tic = time.time()
        # DEFINING PARAMATERS
        # Number of trees in random forest
        n_estimators = n_estimators
        # Number of features to consider at every split
        max_features = max_features
        # Maximum number of levels in tree
        max_depth = max_depth
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = min_samples_split
        # Minimum number of samples required at each leaf node
        min_samples_leaf = min_samples_leaf
        # Method of selecting samples for training each tree
        bootstrap = bootstrap
        
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                       'criterion' :['gini', 'entropy']}
        
        rf_random = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_grid,
                                   n_iter = n_iter, cv = cv, scoring = scoring,
                                   verbose=2, n_jobs = -1)
        
        grid_result = rf_random.fit(X, y)
    
        BestParams_RandomSearch = rf_random.best_params_
        Bestsocre_RandomSearch   = rf_random.best_score_
    
        # summarize results
        
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        print('Randomized search was done in: {} secs'.format(time.time()-tic))
        print("Best: %f using %s" % (Bestsocre_RandomSearch, BestParams_RandomSearch))
        
        return BestParams_RandomSearch, Bestsocre_RandomSearch ,means, stds, params
        #%% Plot feature importance
        
        def Feat_importance_plot(self, Input ,labels, n_estimators = 250):
            classifier = RandomForestClassifier(n_estimators = n_estimators)
            classifier.fit(Input, labels)
            FeatureImportance = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
            sb.barplot(y=FeatureImportance, x=FeatureImportance.index)
            plt.show()
            
        #%% mix pickle and h5 features
        def mix_pickle_h5(self, picklefile, saving_fname,
                          h5file = ("P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/tr90_N3_fp1-M2_fp2-M1.h5"),
                          saving = False, ch = 'fp2-M1'):
            import pickle 
    
            # Define pickle file name
            
            picklefile = picklefile
            pickle_in = open(picklefile + ".pickle","rb")
            Featureset = pickle.load(pickle_in)
            # Open relative h5 file to map labels
            fname = h5file # N3
            ch = ch
            with h5py.File(fname, 'r') as rf:
                xtest  = rf['.']['x_test_' + ch].value
                xtrain = rf['.']['x_train_' + ch].value
                ytest  = rf['.']['y_test_' + ch].value
                ytrain = rf['.']['y_train_' + ch].value
            
            y = np.concatenate((ytrain[:,1], ytest[:,1]))
            
            rp = np.random.permutation(len(y))
            
            X = Featureset[rp,:]
            y = y[rp]
            
            # saving
            if saving == True:
                directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/' 
                fname = saving_fname
                with h5py.File((directory+fname + '.h5'), 'w') as wf:
                    # Accuracies
                    dset = wf.create_dataset('X', X.shape, data =X)
                    dset = wf.create_dataset('y' , y.shape, data  = y)
                    
            return X, y
