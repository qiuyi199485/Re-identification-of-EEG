import os
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
import pandas as pd

##mne settings
mne.set_log_level('WARNING')

def test_edf_corrupted_info(path_to_edf):
    edf_corrupted = False
    edf_info = False
    try:
        f = mne.io.read_raw_edf(
            path_to_edf, 
            preload=False, 
            verbose=0, 
            stim_channel=None)
        edf_info = f.info
        del f
    except:
        print('Import error on file: ' +  path_to_edf)
        edf_corrupted = True
    return (edf_corrupted, edf_info)

def get_date_edf(path_to_edf):
    raw = mne.io.read_raw_edf(
            path_to_edf,
            preload=False, 
            verbose=0, 
            stim_channel=None)
    edf_date=str(raw.info['meas_date'])[0:10]
    return edf_date

def save_df_pkl(df, file_name, save_dir=''):
    if save_dir == '':
        df.to_pickle(file_name + '.pkl')
    else:
        df.to_pickle(save_dir + '/' + file_name + '.pkl')
    
def load_df_pkl(path_to_file):
    df = pd.read_pickle(path_to_file)
    return df
    

def convert_to_pandas_dataframe(dataset_dict):
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