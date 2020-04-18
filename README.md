# ssccoorriinngg
## Introduction
Sleep comprises different stages, namely: N1 (transitional stage), N2 (light sleep), N3 (deep sleep, aka slow wave sleep), and REM (rapid eye movement).

The state of art to detetct various stages of sleep is manual scoring, where an expert looks into different biosignals of human comprising Electroencephalography (EEG), Electrocardiogram (ECG), Electromyography (EMG), and Electrooculography (EOG). However, this process is too time-consuming, effortful, and expensive. This emphasizes the importance of developing automatic algorithms, thanks to the current developments in the artificial inteligence (AI), and machine learning (ML). 

**"ssccoorriinngg"** is an **automatic sleep scoring package** by which you can feed in any EDF file of interest and perform automatic classification. Using this class, one can train any collected sleep data to have his/her own model, or to use the predefined model that we trained using our own dataset and see the classification outputs.

The main aim of this project is to ease sleep analysis and research. So, everyone, having a piece of sleep data in hand (like EEG, ECG, etc) can create his/her own model and try to classify different stages of sleep. Thanks to the existing sleep recording equipment such as headbands (like iBand+) it is easy for any interested person to collect nocturnal or nap sleep data. Then this data can be used to develop a sleep scorer and to analyze the quality of your sleep every day! 

Interesting!

## Description of class
In this section, we define capabilities of **ssccoorriinngg** sleep scorer. This class comprises feature extraction method, feature selection methods, various machine-learning classifiers, and grid/randomized search methods to tune hyper parametrs. The descriptions can be found below. To sue the class please read this page thoroughly.

Since reading EDF files is always time consuming it is always recommended by us to firstly convert them to a more light data format. So, we use the file "EDF_to_h5.py" to perform this conversion and just save the arrays of data and hypnogram (if the aim is training a classifier).

    INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        3) T        : window size (default is 30 secs as in sleep research)
        4) fs       : sampling frequency (Hz)
## 1. FeatureExtraction(): 
This is the main method to extract features and then use the following methods of supervised machine learning algorithms to classify epochs.
    
    INPUTS: 
            It uses global inputs from ML_Depression class, so, doesn't require additional inputs.
        
    OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".

## 2. Classifiers      
        
### 2.1. RandomForest_Modelling( X, y, scoring, n_estimators, cv)
A random forest classifier is made using this function and then a k-fold cross-validation will be used to assess the model classification power.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           scoring      : scoring criteria, e.g. 'accuracy', 'f1_score' etc.
           n_estimator  : number of trees for Random Forest classifier.
           cv           : Cross-validation order
        
    OUTPUTS:
           1) accuracies_RF: Accuracies derived from each fold of cross-validation.
        
### 2.2. KernelSVM_Modelling(X, y, scoring, cv, kernel)

A non-linear SVM model is made using this function and then a k-fold cross-validation will be used to assess the model classification power.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           scoring      : scoring criteria, e.g. 'accuracy', 'f1_score' etc.
           kernel       : kernel function of SVM (e.g. 'db10').
           cv           : Cross-validation order
        
    OUTPUTS:
        1) accuracies_SVM: Accuracies derived from each fold of cross-validation.
### 2.3. LogisticRegression_Modelling(X, y, scroing, cv, max_iter)
A Logistic regression model is made using this function and then a k-fold cross-validation will be used to assess the model classification power.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           scoring      : scoring criteria, e.g. 'accuracy', 'f1_score' etc.
           max_iter     : Maximum number of iterations during training.
           cv           : Cross-validation order
        
    OUTPUTS:
        1) accuracies_LR: Accuracies derived from each fold of cross-validation.
        
### 2.4. XGB_Modelling( X, y, scoring, n_estimators, cv , max_depth, learning_rate):
A XGBoost model (a set of trees, in series) is made using this function and then a k-fold cross-validation will be used to assess the model classification power.
    INPUTS: 
           X              : Featureset input
           y              : Labels (classes)
           scoring        : scoring criteria, e.g. 'accuracy', 'f1_score' etc.
           n_estimators   : # trees
           cv             : Cross-validation order
           max_depth      : Maximum number of levels in tree.
           learning_rate  : the steps taken to learn the data.
    OUTPUTS:
        1) accuracies_xgb : Accuracies derived from each fold of cross-validation.

## 3. Randomized search
This is a method to fine tune the hyper parameters of a model. [sometimes] Randomized search is preferable to Grid search due to Randomly selecting instances among the parameters list, leading to faster computations.

### 3.1. RandomSearchRF(X,y, scoring, estimator, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf,bootstrap,n_iter):

    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
           scoring          : scoring criteria, e.g. 'accuracy', 'f1_score' etc.
           estimator        : RF estimator
           n_estimators     : a list comprising number of trees to investigate.
           max_features     : Number of features to consider at every split.
           max_depth        : Maximum number of levels in tree.
           min_samples_split: Minimum number of samples required to split a node 
           min_samples_leaf : Minimum number of samples required at each leaf node.
           bootstrap        : Method of selecting samples for training each tree.
           n_iter           : The amount of randomly selecting a set of aforementioned parameters.
        
    OUTPUTS:
          1) BestParams_RandomSearch: using 'best_params_' method.
          2) Bestsocre_RandomSearch : using 'best_score_' method.
          
## 4. Feature Selection Methods
Different methods to select the extracted features can be found under this category. There are some filter methods (statistical) in which you need to specify how many features are you willing to select. On the other hand, there are some wrapper (machine-learning-based) methods in which they create ML models themseleves and then select the most represantative features, so there is no need to specify how many features to be selected.

### 4.1. FeatSelect_Boruta(X,y, max_depth):
Boruta method of feature selection. The method creates a random forest / decision tree model and check classification outcomes using different subsets of input features to come up with the most represantative set of features.

    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
    
    OUTPUTS:
           ranks            : ranking of features
           Feat_selected    : new "X" after choosing the most discriminative features.
           
### 4.2. FeatSelect_LASSO(X, y, C)

    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
           C                : Penalization factor
    
    OUTPUTS:
           Feat_selected    : new "X" after choosing the most discriminative features.
           
### 4.3. FeatSelect_ANOVA(X, y, k) 

    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
           k                : number of required features to be selected.
           
    OUTPUTS:
           Feat_selected    : new "X" after choosing the most discriminative features.
           
### 4.4. FeatSelect_Recrusive(X,y,k)
           
    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
           k                : number of required features to be selected.
           
    OUTPUTS:
           Feat_selected    : new "X" after choosing the most discriminative features.

### 4.5. FeatSelect_PCA(X, y, n_components)

    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
           n_components     : number of required principal components.
           
    OUTPUTS:
           Feat_selected    : new "X" after choosing the most discriminative features.
           
### 4.6. Feat_importance_plot(Input ,labels, n_estimators)
Plots feature importance for Random Forest model.

    INPUTS: 
           Input        : Featureset input
           labels       : Labels (classes)
           n_estimators : number of trees in RF

           
## 5. Others:
This category belong to other methods of the ML_Depression class.

### 5.1. SaveFeatureSet(X, y, path, filename)
This method is to save the extracted featrues.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           path         : Location to save (final FOLDER)
           filename     : FILENAME to save in "path"
           cv           : Cross-validation order
           
### 5.2. LoadFeatureSet( path, fname, feats, labels)  

    INPUTS: 
           path         : Location to save (final FOLDER)
           fname        : FILENAME to save in "path"
           feats        : Featureset input 
           labels       : Labels (classes)
           
### 5.3. CombineEpochs(directory, ch, N3_fname, REM_fname , saving = False, fname_save)
This funcyion can combine different stages together. The main interest was to combine SWS (N3) and REM stages for classification.

    INPUTS: 
           directrory   : Location to load files (final FOLDER)
           N3_fname     : FILENAME of N3 stages (.h5 file)
           REM_fname    : FILENAME of REM stages (.h5 file)
           saving       : to save results: True
           fname_save   : name of file to save results
           
## Sample code to use methods
to see examples of using code, see example folder.

