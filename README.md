# ssccoorriinngg : Automatic sleep scoring package
![ssccoorriinngg_logo](https://user-images.githubusercontent.com/48684369/82963192-78f4c380-9fc2-11ea-8412-f0a16f02c6ed.png)
## Introduction
Sleep comprises different stages, namely: N1 (transitional stage), N2 (light sleep), N3 (deep sleep, aka slow-wave sleep), and REM (rapid eye movement).

The state of art to detect various stages of sleep is manual scoring, where an expert looks into different biosignals of a human comprising Electroencephalography (EEG), Electrocardiogram (ECG), Electromyography (EMG), and Electrooculography (EOG). However, this process is too time-consuming, effortful, and expensive. This emphasizes the importance of developing automatic algorithms, thanks to the current developments in the artificial intelligence (AI), and machine learning (ML). 

**"ssccoorriinngg"** is an **automatic sleep scoring package** by which you can feed in any EDF file of interest and perform automatic classification. Using this class, one can train any collected sleep data to have his/her own model, or to use the predefined model that we trained using our own dataset and see the classification outputs.

The main aim of this project is to ease sleep analysis and research. So, everyone, having a piece of sleep data in hand (like EEG, ECG, etc) can create his/her own model and try to classify different stages of sleep. Thanks to the existing sleep recording equipment such as headbands (like iBand+) it is easy for any interested person to collect nocturnal or nap sleep data. Then this data can be used to develop a sleep scorer and to analyze the quality of your sleep every day! 

**ssccoorriinngg** goal is not to provide the most complicated model for sleep scoring, but simpler models with comparable efficiency! Therefore, we would like the models to be quite useful for any minimal sensing system, e.g. sleep headbands.

Interesting!

![hyp_subject85](https://user-images.githubusercontent.com/48684369/83881068-686de700-a740-11ea-9fb1-814d44508866.png)
## Description of class
In this section, we define the capabilities of **ssccoorriinngg** sleep scorer. This class comprises feature extraction method, feature selection methods, various machine-learning classifiers, and grid/randomized search methods to tune hyper parameters. The descriptions can be found below. To sue the class please read this page thoroughly.

Since reading EDF files is always time-consuming it is always recommended by us to first convert them to a more light data format. So, we use the file "EDF_to_h5.py" to perform this conversion and just save the arrays of data and hypnogram (if the aim is training a classifier).

    INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        3) T        : window size (default is 30 secs as in sleep research)
        4) fs       : sampling frequency (Hz)
## 1.1 FeatureExtraction(): 
This is the main method to extract features and then use the following methods of supervised machine learning algorithms to classify epochs. * This method combines the whole existing data which is fed in, extract features of each observation and randomly permute them in the end.*
    
    INPUTS: 
            It uses global inputs from ML_Depression class, so, doesn't require additional inputs.
        
    OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".

## 1.2 FeatureExtraction_per_subject(Input_data): 
This is the main method to extract features and then use the following methods of supervised machine learning algorithms to classify epochs. * This method is useful to see the hypnogram and featuresets PER SUBJECT or per trial.*
    
    INPUTS: 
        1) Input_data: is the .h5 file which hast to be already loaded.
        
    OUTPUTS:
    Outputs are in the form of dictionary.
        1) Featureset        : Concatenation of all featureset after random permutation.


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
### 5.4. remove_bad_signals(hypno_labels, input_feats)           
This function is to remove the labels and data from the channels which have already been marked as disconnection or bad signal during manual scoring:

    INPUTS: 
           input_feats      : Featureset input
           hypno_labels     : hypongram labels
           
    OUTPUTS:
           out_feats        : new Featureset.
           out_labels       : new labels.
           
### 5.5. remove_arousals_and_wake(hypno_labels, input_feats)

    INPUTS: 
           input_feats      : Featureset input
           hypno_labels     : hypongram labels
           
    OUTPUTS:
           out_feats        : new Featureset.
           out_labels       : new labels.
           
### 5.6. replace_arousal_with_wake(hypno_labels, input_feats)

    INPUTS: 
           input_feats      : Featureset input
           hypno_labels     : hypongram labels
           
    OUTPUTS:
           out_labels       : new labels.

### 5.7. binary_labels_creator(labels)

    INPUTS: 
           labels           : hypongram labels
           
    OUTPUTS:
           out_labels       : output binary labels.

### 5.8. save_dictionary(path, fname, labels_dic, features_dic)

    INPUTS: 
           path             : Final folder of saving data.
           fname            : name of file to save.
           labels_dic       : Array including hypongram labels.
           features_dic     : Array comprising featureset.          

### 5.9. load_dictionary(path, fname)

    INPUTS: 
           path             : Final folder of saving data.
           fname            : name of file to save.
           
    OUTPUTS:
           feats            : loaded features
           y                : output binary labels.

### 5.10. Standardadize_features(X_train, X_test)

    INPUTS: 
           X_train          : Featureset input
           y_train          : Labels (classes)
           
    OUTPUT:
           
           X_test           : Featureset input
           y_test           : Labels (classes)
           
### 5.11. replace_NaN_with_mean(Features)

    INPUTS:
           Features            : Input features
    OUTPUTS:
           Features            : Features without NaN
           
### 5.12. replace_inf_with_mean(Features)

    INPUTS:
           Features            : Input features
    OUTPUTS:
           Features            : Features without inf
      
 ### 5.13.  binary_to_single_column_label(y_pred)     
 
    INPUTS:
           y_pred              : binaty labels (each class is a column and each row is an observation)
    OUTPUTS:
           hyp_pred            : Output as one column (wake:0, N1:1, N2:2, N3:3, REM: 4)
           
### 5.14. plot_hyp(hyp, mark_REM = 'active')

    INPUTS:
           hyp                 : Hypnogram in the form of 1 column with different values.
           mark_REM            : if = 'active', REM periods are marked in red.
           
    OUTPUTS:
           Plot_hyp            : Hypnogram plot

### 5.15. plot_comparative_hyp(hyp_true, hyp_pred, mark_REM = 'active'):

    INPUTS:
           hyp_true            : Hypnogram (TRUE VALUES) in the form of 1 column with different values.
           hyp_pred            : Hypnogram (PREDICTED VALUES) in the form of 1 column with different values.
           mark_REM            : if = 'active', REM periods are marked in red.
           
    OUTPUTS:
           Plot_hyp            : Hypnogram plot

### 5.16. Mean_leaveOneOut(metrics_per_fold) 

    INPUTS:
           metrics_pre_fold    : Outcome metrics of each fold as a dictionary.
           
    OUTPUTS:
           print               : Mean values of all metrics.
           
### 5.17. save_figure(directory, saving_name, dpi, saving_format = '.png',full_screen = False)

    INPUTS: 
           directory    : Location to save (final FOLDER)
           saving_name  : FILENAME to save in "path"
           dpi          : sets quality of saved image
           saving_format: format, can be anything, e.g. .png, .jpg. etc
           cv           : Cross-validation order
           
### 5.18. add_time_dependence_to_features(featureset, n_time_dependence=3)   

    INPUTS:
           featurest           : featureset which is already made.
           n_time_dependence   : number of epochs (forward and  backward) to consider as input for the current epoch.
           
    OUTPUTS:
           X_new               : New featureset
           
## Sample code to use methods
to see examples of using code, see example folder.

