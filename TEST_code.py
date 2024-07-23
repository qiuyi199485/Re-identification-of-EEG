import os
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
import pandas as pd
import numpy as np
##mne settings
mne.set_log_level('WARNING')

def test_edf_corrupted_info(path_to_edf):                      ##check if xx.edf can be read
    edf_corrupted = False
    edf_info = False
    try:
        f = mne.io.read_raw_edf(                                ## mne.io.f = mne对象， 包含了edf的元数据信息metadata
            path_to_edf, 
            preload=False,                                      
            verbose=0, 
            stim_channel=None)
        edf_info = f.info 
        edf_time = int(f.times[-1]) 
        edf_ch_names = f.info['ch_names']## edf可以用mne读出来就是 edf_info=edf所有metadat   
        del f
    except:                                                     ## error
        print('Import error on file: ' +  path_to_edf)
        edf_corrupted = True
        
    return (edf_corrupted, edf_info, edf_time,edf_ch_names)



path = 'C:\\Users\\49152\\Desktop\\MA\\Code\\000\\aaaaaaaj\\s001_2002\\02_tcp_le\\aaaaaaaj_s001_t000.edf'
patients,a,b,c = test_edf_corrupted_info(path)
print(a)
print(c)
print(len(c))

required_channels = {"EEG FP1-LE", "EEG FP2-LE", "EEG F7-LE", "EEG F3-LE", "EEG FZ-LE", "EEG F4-LE", "EEG F8-LE", 
                         "EEG A1-LE", "EEG T3-LE", "EEG C3-LE", "EEG CZ-LE", "EEG C4-LE", "EEG T4-LE", "EEG A2-LE", 
                         "EEG T5-LE", "EEG P3-LE", "EEG PZ-LE", "EEG P4-LE", "EEG T6-LE", "EEG O1-LE", "EEG O2-LE"}


channels_set = set(c)
for ch in c:
    if ch in required_channels:
        print(f"{ch} 是指定的通道")
    else:
        print(f"{ch} 不是指定的通道")
# 判断是否包含所有指定的通道名称
if required_channels.issubset(channels_set):
    print("都包含")
else:
    print("没有都包含")
    missing_channels = target_set - channels_set
    print("缺少的通道: ", missing_channels)