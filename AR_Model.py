#Import created script

import sys                                                                       
import os
os.environ['MNE_USE_NUMBA'] = 'false'                                            #避免使用numba speedup
import mne
import numpy as np
from IPython.display import clear_output
import pandas as pd


sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')                    # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
import settings
from tools import test_edf_corrupted_info, get_date_edf                      ## (edf_corrupted, edf_info) false没坏，edf_metadata; edf measurment date yyyy-mm-dd

from statsmodels.tsa.ar_model import AutoReg


# Read EEG data 
def read_eeg_from_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    eeg_data = raw.get_data()
    return eeg_data


# Fit AR model 
def fit_ar_model(eeg_data, lags=10):
    ar_features = []
    for channel_data in eeg_data:
        model = AutoReg(channel_data, lags=lags)
        model_fit = model.fit()
        ar_features.append(model_fit.params)
    ar_features = np.array(ar_features)
    return ar_features

# Extract AR features
def process_and_extract_features(patients):
    features_dict = {}
    for patient_id, records in patients.items():
        features_dict[patient_id] = []
        for record in records:
            date_of_the_take, session_id_take, path_to_edf, edf_info = record
            eeg_data = read_eeg_from_edf(path_to_edf)
            ar_features = fit_ar_model(eeg_data)
            features_dict[patient_id].append((session_id_take, ar_features))
    return features_dict

# Example usage
path = 'your_dataset_path'
patients = get_patients(path)
features_dict = process_and_extract_features(patients)

# Convert the extracted features to a DataFrame
def convert_features_to_dataframe(features_dict):
    feature_list = []
    for patient_id, records in features_dict.items():
        for record in records:
            session_id, features = record
            feature_list.append([patient_id, session_id, features])
    df = pd.DataFrame(feature_list, columns=['patient_id', 'session_id', 'ar_features'])
    return df

features_df = convert_features_to_dataframe(features_dict)

# Display the DataFrame with AR features
print(features_df)
