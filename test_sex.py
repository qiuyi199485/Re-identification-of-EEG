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

import autoreject


path_to_edf = 'C:\\Users\\49152\\Desktop\\MA\\Code\\000\\aaaaaaaa\\s001_2015\\01_tcp_ar\\aaaaaaaa_s001_t000.edf' 

raw = mne.io.read_raw_edf(                                  ## mne library function for edf; mne.io.raw = mne对象， 包含了edf的元数据信息metadata
            path_to_edf,                                        ## input path
            preload=False,                                      ## save memory ; 数据在需要时才从磁盘读取
            verbose=0,                                          ## 函数运行时，输出的详细信息级别
            stim_channel=None) 


def filter_freq(raw, freq = 50, f_min=st.f_min, f_max=st.f_max):                            #频率过滤
    filter_raw = raw.copy().load_data().filter(f_min, f_max, fir_design='firwin')           # 复制raw数据，保留原raw信号
    filter_raw = filter_raw.notch_filter(freq)                                              # freq 陷波频率 一般50Hz 去除50HZ影响
    
    return filter_raw

#Xtest = np.load('C:\\Users\\49152\\Desktop\\MA\\Code\\000\\aaaaaaaa\\s001_2015\\01_tcp_ar\\aaaaaaaa_s001_t000.edf') 
# shape: (num_eeg_segments, num_channels=21, time (e.g. 4 seconds at 250 Hz $\hat{=}$ 1000), 1)



def read_eeg_data_from_file_to_np(
        file_path: str, 
        channels, 
        duration: float, 
        start: float, 
        lower_freq, 
        upper_freq) -> np.array:
    """Return a NumPy array with the signal data, where each row represents a channel and each column a time sample.
    
    
    Args:
        file_path (str): [description]
        channels (array_like): [description]
        duration (float): duration of recording in minutes
        start (float): start time in minutes
        lower_freq (float, optional): [description]. Defaults to 0.5.
        upper_freq (int, optional): [description]. Defaults to 40.

    Returns:
        np.array: [description]
    """
    f = mne.io.read_raw_edf(file_path, verbose=0)
    f.crop(start,
           start+duration)
    f.load_data()
    f.filter(lower_freq, upper_freq)
    assert f.info['sfreq']==st.sfreq
    # Get numpy array
    data = f.get_data(
        picks=channels, 
        reject_by_annotation=None, 
        return_times=False)
    
    return data



filter_raw=filter_freq(raw)

print(filter_raw.info)    
 
#y_pred = model.predict(Xtest, batch_size=1)


# Load trained Model for sex and age

model_path = 'C:\\Users\\49152\\Desktop\\MA\\Code\\pretrained_net_ica1-40Hz'
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'K': K},
        compile=False)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")