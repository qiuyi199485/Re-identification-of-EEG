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

Xtest = np.load('C:\\Users\\49152\\Desktop\\MA\\Code\\000\\aaaaaaaa\\s001_2015\\01_tcp_ar\\aaaaaaaa_s001_t000.edf') 
# shape: (num_eeg_segments, num_channels=21, time (e.g. 4 seconds at 250 Hz $\hat{=}$ 1000), 1)

     
y_pred = model.predict(Xtest, batch_size=1)
#gender, age = predict_from_edf(path_to_edf_files)     