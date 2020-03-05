# Random Forest Classification - Template code by Mahdad
''' PLEASE NOTE: 
    -You need to make some changes to make this code work!
    -In case you cannot import a certain library it should be installed on your
     pc. 
    -Dataset name should be addressed in STEP 1.1.
    -Fill in "?" signs by correct values.
    -This code is made such that can be used for many different algorithms. The
     only section you need to change for using another machine learning method 
     is "STEP 2". Most of the rest of steps remain identical.
    '''
## STEP 0: Importing the libraries ##
import numpy as np                     
import matplotlib.pyplot as plt            
import pandas as pd

## STEP 1.1:Importing the dataset ##
fname = 'Social_Network_Ads.csv' # ADD DATASET NAME HERE!
dataset = pd.read_csv(fname)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# STEP 1.2: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)

# STEP 1.3: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# STEP 2:Fitting Random Forest Classification to the Training set 
'''(push "Ctrl+i" on # RandomForestClassifier to see the documentation of variables.)'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion ='gini', random_state = 0)
classifier.fit(X_train, y_train)

# STEP 3.1: Predicting the Test set results
y_pred = classifier.predict(X_test)

# STEP 3.2: Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# STEP 4: Define accuracy based on CM components
Acc = (cm[0,0]+cm[1,1]) / np.sum(cm)

###################### EXERCICES ##############################

### Ex1 ### Try to tune the model such that you get the best accuracy ###
# HINT: play with parameters of model, e.g. max_depth, n_estimator, etc.

### ٍEx2 ### Create a decision tree model instead of random forest and compare the results 
# HINT: STEP2: 
from sklearn.tree import DecisionTreeClassifier

### Ex3 ###: How can we include a categorical feature in the training set?
# HINT: in step 1.3 add: 
from sklearn.preprocessing import OneHotEncoder

### Ex4 ### Write a line to evaluate the performance of model using cross-validation.
# HINT: 
from sklearn.model_selection import cross_val_score

### Ex5 ### Apply randomzied grid search to find the best parameters of model.
# HINT:
from sklearn.model_selection import RandomizedSearchCV

### Ex6 ### Apply Grid Search and compare it with randomzied grid.
# What is the difference between them?

################ ADVANCED EXERCISE: GRAB YOUR OWN DATA! ##################

# 1. Take an EEG data and its labels from your own study and copy to the same 
# directory as this ".py" file.
# 2. Apply pre-processing (see MNE toolbox)
# 3. What features are relevant for sleep stage classification? Try to extract them.
# HINT: Start with power spectrum in different frequency ranges and some statistical
# features such as skewness, kurtosis, etc.
# 4.Apply a random forest classifier and try to classify your data by machine learning!

