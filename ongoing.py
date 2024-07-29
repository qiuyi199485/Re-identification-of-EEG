## 提取EEG特征
import os
import sys
import mne
import numpy as np
import pandas as pd
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
print(features_df)

##SVM分类器

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Prepare the data for classification
X = np.vstack(features_df['ar_features'].values)
y = features_df['patient_id'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the classifier model
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Classification Accuracy: {accuracy}")

## 比较相似度
from scipy.spatial.distance import cosine, euclidean

# Extract features for EEG signal "A"
eeg_data_A = read_eeg_from_edf(path_to_eeg_A)
features_A = fit_ar_model(eeg_data_A)

# Calculate similarity with other signals
similarities = []
for index, row in features_df.iterrows():
    patient_id = row['patient_id']
    session_id = row['session_id']
    features = row['ar_features']
    similarity = 1 - cosine(features_A.flatten(), features.flatten())  # 余弦相似性
    # similarity = euclidean(features_A.flatten(), features.flatten())  # 欧氏距离
    similarities.append((patient_id, session_id, similarity))

# Sort by similarity
similarities.sort(key=lambda x: x[2], reverse=True)

# Display the most similar signals
similar_signals_df = pd.DataFrame(similarities, columns=['patient_id', 'session_id', 'similarity'])
print(similar_signals_df.head())
