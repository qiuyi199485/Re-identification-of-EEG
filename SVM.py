import sys                                                                       
import os
os.environ['MNE_USE_NUMBA'] = 'false'                                            #避免使用numba speedup
import mne
import numpy as np
from IPython.display import clear_output
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.spatial.distance import euclidean

sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')  # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
import settings
from tools import test_edf_corrupted_info, get_date_edf
from getDataset import get_patients   
from statsmodels.tsa.ar_model import AutoReg

# Read EEG data 
def read_eeg_from_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    eeg_data = raw.get_data()
    return eeg_data

# Fit AR model 
def fit_ar_model(eeg_data, lags=6):
    ar_features = []
    for channel_data in eeg_data:
        model = AutoReg(channel_data, lags=lags)
        model_fit = model.fit()
        ar_features.append(model_fit.params)
    ar_features = np.array(ar_features)
    return ar_features

# Extract AR features
def process_and_extract_features(patients, max_patients=100):
    features_dict = {}
    patient_count = 0
    for patient_id, records in patients.items():
        if patient_count >= max_patients:
            break
        features_dict[patient_id] = []
        for record in records:
            date_of_the_take, session_id_take, path_to_edf, edf_info = record
            eeg_data = read_eeg_from_edf(path_to_edf)
            ar_features = fit_ar_model(eeg_data)
            features_dict[patient_id].append((session_id_take, ar_features))
        patient_count += 1
    return features_dict

# Example 
path = 'C:\\Users\\49152\\Desktop\\MA\\Code'
patients = get_patients(path)
features_dict = process_and_extract_features(patients, max_patients=100)

# Features to DataFrame
def convert_features_to_dataframe(features_dict):
    feature_list = []
    for patient_id, records in features_dict.items():
        for record in records:
            session_id, features = record
            feature_list.append([patient_id, session_id, features])
    df = pd.DataFrame(feature_list, columns=['patient_id', 'session_id', 'ar_features'])
    return df

features_df = convert_features_to_dataframe(features_dict)

# Prepare data for classification
X = []
y = []
for _, row in features_df.iterrows():
    for feature in row['ar_features']:
        X.append(feature.flatten())
        y.append(row['patient_id'])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classifier
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Individual identification
def find_similar_signal(features_df, target_feature):
    distances = []
    for _, row in features_df.iterrows():
        for feature in row['ar_features']:
            distance = euclidean(target_feature.flatten(), feature.flatten())
            distances.append((distance, row['patient_id'], row['session_id']))
    distances.sort(key=lambda x: x[0])
    return distances

# Assuming A's features are in `target_feature`
target_patient_id = 'example_patient_id'  # replace with actual patient_id
target_session_id = 'example_session_id'  # replace with actual session_id
target_feature = features_df[(features_df['patient_id'] == target_patient_id) & 
                             (features_df['session_id'] == target_session_id)]['ar_features'].values[0]

similar_signals = find_similar_signal(features_df, target_feature)
print(f'Top 5 similar signals to A: {similar_signals[:5]}')
