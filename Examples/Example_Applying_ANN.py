# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:10:58 2020

@author: mahda
"""
#%% Load libraries
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from ssccoorriinngg import ssccoorriinngg

#%% Picking featureset of interest and apply classification
Object = ssccoorriinngg(filename='', channel='', fs = 200, T = 30)
path   = 'C:/PhD/ML in depression/'
fname  = 'feat42_Fp1-Fp2_train'
feats  = 'featureset'
labels = 'labels'
X, y   = Object.LoadFeatureSet(path, fname, feats, labels)

#%% Creating ANN Model
# Set random seed
np.random.seed(0)

# Number of features
number_of_features = np.shape(X)[1]

# Create function returning a compiled network
def create_network(units2=20,unit1=16):
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=unit1, activation='relu', input_shape=(np.shape(X)[1],)))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=units2, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=4, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='crossentropy', # Cross-entropy
                    optimizer='adam', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network

#%% Apply 10-fold Cross-validation
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision': make_scorer(precision_score),
           'recall  ' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}  

# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=10, 
                                 batch_size=100, 
                                 verbose=2)

# Evaluate neural network using three-fold cross-validation
results_ANN = cross_validate(neural_network, X, y, cv=10, scoring = scoring)


#%% Outcome measures
# Defien required metrics here:
Metrics = ['test_accuracy', 'test_precision', 'test_recall  ', 'test_f1_score']
for metric in Metrics:

    r1      = results_ANN[metric].mean()
    std1    = results_ANN[metric].std()
    print(f'{metric} for RF is: {round(r1*100, 2)}+- {round(std1*100, 2)}')
