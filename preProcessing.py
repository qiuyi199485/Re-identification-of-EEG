# import f
import sys
import os
import numpy as np
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap, )
import tensorflow as tf
from tensorflow.keras import backend as K

sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')  
import settings as st
import Build_Database as bd


    
def filter_freq(raw, freq = [], f_min=st.f_min, f_max=st.f_max):                            #频率过滤
    filter_raw = raw.copy().load_data().filter(f_min, f_max, fir_design='firwin')           # 复制raw数据，保留原raw信号
    filter_raw = filter_raw.notch_filter(freq)                                              # freq 陷波频率 一般50Hz 去除50HZ影响
    
    return filter_raw




def create_classification(df_train, df_val, df_test):                                       # 数据基于Patient_id分类处理
    #get all patients id from the dataframes                                                #来自Build_Database的 df 选取第一列Patient_id
    train_pat_id, test_pat_id, val_pat_id = df_train.values[:,0], df_test.values[:,0], df_val.values[:,0]
    
    #empty arrays for the classifications of the dataframes 空数组用于分类
    train_pat_clas, test_pat_clas, val_pat_clas = [], [], []
    
    train_pat_unique = np.unique(train_pat_id)               # 选一个训练集中只有一个.edf的Patient的id
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
    for patient_id in train_pat_id:                                     # 给每个训练集的病人一个分类 
        index_list = np.nonzero(train_pat_unique == patient_id)[0]
        train_pat_clas.append(train_pat_unique_clas[index_list[0]])
    
    #every patient from the test dataframe should gets classified (or classified as uncalssified)
    for patient_id in test_pat_id:
        index_list = np.nonzero(train_pat_unique == patient_id)[0]
        if index_list.size == 0:                   # 给每个测试集的病人根据训练集分完的分类，也有可能在训练集里面没出现，那之后就要删除！
            test_pat_clas.append(general_clas)
        else:
            index = index_list[0]
            test_pat_clas.append(train_pat_unique_clas[index])
    
    
    #every patient from the validation dataframe gets classified (or classified as uncalssified)
    for patient_id in val_pat_id:
        index_list = np.nonzero(train_pat_unique == patient_id)[0]
        if index_list.size == 0:                    # 给每个验证集的病人根据训练集分完的分类，也有可能在训练集里面没出现，那之后就要删除！
            val_pat_clas.append(general_clas)
        else:
            index = index_list[0]
            val_pat_clas.append(train_pat_unique_clas[index])
            
            
    # add the classifiication to the main dataframe train, test and validation  在三个df的最后，也就是第六列一列分类标签 
    # 根据index来标签，例如第一个病人(1,0,0,0...) 第二个病人 (0,1,0,0...) ,第3个病人 (0,0,1,0...) 
    df_train.insert(df_train.shape[-1], "classification", train_pat_clas)
    df_test.insert(df_test.shape[-1], "classification", test_pat_clas)
    df_val.insert(df_val.shape[-1], "classification", val_pat_clas)
          
    
    return df_train, df_val, df_test
    

df_train, df_val, df_test=create_classification(bd.train_dataset_pd, bd.validation_dataset_pd, bd.test_dataset_pd)      

                      