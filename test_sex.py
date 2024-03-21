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

def filter_freq(raw, freq = 50, f_min=st.f_min, f_max=st.f_max):                            #频率过滤
    filter_raw = raw.copy().load_data().filter(f_min, f_max, fir_design='firwin')           # 复制raw数据，保留原raw信号
    filter_raw = filter_raw.notch_filter(freq)                                              # freq 陷波频率 一般50Hz 去除50HZ影响
    
    return filter_raw



channels = ['EEG FP1-REF', 'EEG FP2-REF','EEG F7-REF','EEG F3-REF','EEG FZ-REF','EEG F4-REF','EEG F8-REF','EEG A1-REF','EEG T3-REF',
            'EEG C3-REF','EEG CZ-REF','EEG C4-REF','EEG T4-REF','EEG A2-REF','EEG T5-REF','EEG P3-REF','EEG PZ-REF','EEG P4-REF',
            'EEG T6-REF','EEG O1-REF','EEG O2-REF']

channels_of_interest = ['EEG FP1-REF', 'EEG FP2-REF','EEG F7-REF','EEG F3-REF','EEG FZ-REF','EEG F4-REF','EEG F8-REF','EEG A1-REF','EEG T3-REF',
            'EEG C3-REF','EEG CZ-REF','EEG C4-REF','EEG T4-REF','EEG A2-REF','EEG T5-REF','EEG P3-REF','EEG PZ-REF','EEG P4-REF',
            'EEG T6-REF','EEG O1-REF','EEG O2-REF']

def read_eeg_data_from_file_to_np(
        path_to_edf: str, 
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
    f = mne.io.read_raw_edf( 
            path_to_edf,                                        ## input path
            preload=False,                                      ## save memory ; 数据在需要时才从磁盘读取
            verbose=0,                                          ## 函数运行时，输出的详细信息级别
            stim_channel=None)
    
    
    
    f.crop(start,
           start+duration)
    f.load_data()
    f.filter(lower_freq, upper_freq)
    #assert f.info['sfreq']==st.f_s
    if f.info['sfreq'] != st.f_s:
     print(f"Resample to 250 Hz")
     f.resample(sfreq=st.f_s, npad="auto")
    
    f.set_eeg_reference(ref_channels='average')
    # Get numpy array
    data = f.get_data(
        picks=channels, 
        reject_by_annotation=None, 
        return_times=False)
    
    return data




def run_autoreject_pipeline(
    path_to_edf, 
    channels_of_interest,
    tmax_sec,
    window_size,
    overlap, 
    tmin_sec):
    """Loads *.edf and returns global threshold. 

    Args:
        file_path (str): Path to edf file.
        channels_of_interest (list): List of channel names
        tmax_sec (float): Last EEG time-point in seconds.
        window_size (int, optional)
        overlap (float, optional) 
        tmin_sec (int, optional)

    Returns:
        float: Threshold
    """
    raw = mne.io.read_raw_edf( 
            path_to_edf,                                        ## input path
            preload=False,                                      ## save memory ; 数据在需要时才从磁盘读取
            verbose=0,                                          ## 函数运行时，输出的详细信息级别
            stim_channel=None)
    
    raw = raw.copy().pick_channels(channels_of_interest, ordered=True)
    raw.crop(tmin_sec,tmax=tmax_sec)
    raw.load_data()
    raw.filter(1,40)
    raw.set_eeg_reference(ref_channels='average')
    #return loop_autoreject(raw, n_splits=5, window_size=window_size, overlap=overlap)


data = read_eeg_data_from_file_to_np(path_to_edf,channels,4/60,2,1,40)
 
 
 
#


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
    
#Xtest = np.load('C:\\Users\\49152\\Desktop\\MA\\Code\\000\\aaaaaaaa\\s001_2015\\01_tcp_ar\\aaaaaaaa_s001_t000.edf') 
# shape: (num_eeg_segments, num_channels=21, time (e.g. 4 seconds at 250 Hz $\hat{=}$ 1000), 1)    
Xtest =  data  
y_pred = model.predict(Xtest, batch_size=1)