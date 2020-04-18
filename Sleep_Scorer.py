# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:39:54 2020

@author: mahjaf

"""
#%% Importing libs
import numpy as np
from   numpy import loadtxt
import h5py
import time
import os 

# Distinguishing patients from control group
gp = loadtxt("P:/3013080.02/ml_project/grouping.txt", delimiter="\t", skiprows = 1, dtype = 'str')
subj_c = [] # Control
subj_p = [] # Patients
# Find control subject IDs
for indx, c in enumerate(gp):
    if c[1] == 'C':
        subj_c.append(int(c[0]))
    elif c[1] == 'CC':
        pass
    else:
        subj_p.append(int(c[0]))
# Initialization
subjects_dic = {}
hyp_dic      = {}
#%% Read data per subject and assign it to relevant array
for idx, c_subj in enumerate(subj_c):
    print (f'Analyzing Subject Number: {c_subj}')
    tic = time.time()
    path = 
    with h5py.File(path +'LK_'+ c_subj + '_1.h5', 'r') as rf:
    x_tmp  = rf['.'][data_fp1-fp2].value
    y_tmp  = rf['.'][hypnogram].value
    ########### HERE UI SHOULD FIRSTLY EXTRACT FEATS AND THEN PUT IN DIC. 
    # ALSO IRRELAVNT LABELS SHOULD BE REMOVED
    subjects_dic["subject{}".format(c_subj)] = x_tmp
    hyp_dic["hyp{}".format(c_subj)] = y_tmp
    del x_tmp, y_tmp
    toc = time.time()
    print(f'Features of subject {c_subj} were successfully extracted in: {toc-tic} secs')
#%% Create leave-one-out cross-validation

    
