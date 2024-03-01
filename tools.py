import os
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
import pandas as pd

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
        edf_info = f.info                                       ## edf可以用mne读出来就是 edf_info=edf所有metadata
        del f
    except:                                                     ## error
        print('Import error on file: ' +  path_to_edf)
        edf_corrupted = True
    return (edf_corrupted, edf_info)

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
    

def convert_to_pandas_dataframe(dataset_dict):                   ## dataset dic --> panda dataframe , include 
    convert_list = []
    keys = dataset_dict.keys()
    for key in keys:
        patient_id = str(key)
        takes = dataset_dict[key] 
        for take in takes:
            session_id, take_id, path_to_edf, info_meta = take
            convert_list.append([patient_id, session_id, take_id, path_to_edf, info_meta])
    df = pd.DataFrame(np.array(convert_list), columns=['patient_id', 'session_id/date', 'take_id', 'path_to_edf', 'edf_info'])
    
    return df