import sys
import os
import numpy as np
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap, )
sys.path.insert(1, 'AVATAR/code/mphilipp/')
import settings
import getDataset as gd

    
def filter_freq(raw, freq = [], f_min=settings.f_min, f_max=settings.f_max):
    filter_raw = raw.copy().load_data().filter(f_min, f_max, fir_design='firwin')
    filter_raw = filter_raw.notch_filter(freq)
    
    return filter_raw


def create_classification(df_train, df_val, df_test):
    #get all patients id from the dataframes
    train_pat, test_pat, val_pat = df_train.values[:,0], df_test.values[:,0], df_val.values[:,0]
    
    #empty arrays for the classifications of the dataframes
    train_pat_clas, test_pat_clas, val_pat_clas = [], [], []
    
    train_pat_unique = np.unique(train_pat)
    train_pat_unique_clas = []
    #get a general calssifiction, which mean that no of the train patients is classified
    general_clas = np.zeros(train_pat_unique.size)
    
    #classify the unique train patients, all gets an class, this are the classes we want to classify at the end
    for i in range(0, train_pat_unique.size):
        pat_clas = np.copy(general_clas)
        pat_clas[i] = 1
        train_pat_unique_clas.append(pat_clas)
        del pat_clas

    #every patient from the train dataframe gets classified
    for patient_id in train_pat:
        index_list = np.nonzero(train_pat_unique == patient_id)[0]
        train_pat_clas.append(train_pat_unique_clas[index_list[0]])
    
    #every patient from the test dataframe should gets classified (or classified as uncalssified)
    for patient_id in test_pat:
        index_list = np.nonzero(train_pat_unique == patient_id)[0]
        if index_list.size == 0:
            test_pat_clas.append(general_clas)
        else:
            index = index_list[0]
            test_pat_clas.append(train_pat_unique_clas[index])
    
    
    #every patient from the validation dataframe gets classified (or classified as uncalssified)
    for patient_id in val_pat:
        index_list = np.nonzero(train_pat_unique == patient_id)[0]
        if index_list.size == 0:
            val_pat_clas.append(general_clas)
        else:
            index = index_list[0]
            val_pat_clas.append(train_pat_unique_clas[index])
            
            
    # add the classifiication to the main dataframe train, test and validation
    
    df_train.insert(df_train.shape[-1], "classification", train_pat_clas)
    df_test.insert(df_test.shape[-1], "classification", test_pat_clas)
    df_val.insert(df_val.shape[-1], "classification", val_pat_clas)
          
    
    return df_train, df_val, df_test
    

