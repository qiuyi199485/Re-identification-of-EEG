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
        
    return (edf_corrupted, edf_info, edf_time, edf_ch_names)

def get_date_edf(path_to_edf):                                  ## EEG data
    raw = mne.io.read_raw_edf(                                  ## mne library function for edf; mne.io.raw = mne对象， 包含了edf的元数据信息metadata
            path_to_edf,                                        ## input path
            preload=False,                                      ## save memory ; 数据在需要时才从磁盘读取
            verbose=0,                                          ## 函数运行时，输出的详细信息级别
            stim_channel=None)                                  ## channel：   指定刺激通道
    edf_date=str(raw.info['meas_date'])[0:10]                   ## edf_data=读取了edf的测试时间信息(字符串string)！‘meas_date’; [0:10]=选取string 前十 一般为yyyy-mm-dd
    return edf_date

def save_df_pkl(df, file_name, save_dir=''):                   ## pickle= Data persistence preservation ; data~ bytes string     让数据持久化保存，但不可读
    if save_dir == '':
        df.to_pickle(file_name + '.pkl')
    else:
        df.to_pickle(save_dir + '/' + file_name + '.pkl')
    
def load_df_pkl(path_to_file):
    df = pd.read_pickle(path_to_file)
    return df
    

def convert_to_pandas_dataframe(dataset_dict):
    convert_list = []
    keys = dataset_dict.keys()                                       # P_id=Patient id 'aaaaaaac'
    for P_id in keys:
        patient_id = str(P_id)                                       # P_record
        P_record = dataset_dict[P_id] 
        for take in P_record:
            session_id, take_id, session_time,session_date,path_to_edf, info_meta = take
            convert_list.append([patient_id, session_id, take_id, session_time,session_date,path_to_edf, info_meta])
    df = pd.DataFrame(np.array(convert_list), columns=['patient_id', 'session_id', 'token_id', 'edf_time','session_date','path_to_edf', 'edf_info'])
    
    return df

