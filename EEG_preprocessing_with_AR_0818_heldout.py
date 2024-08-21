import matplotlib.pyplot as plt
import os
import sys
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from autoreject import AutoReject

sys.path.insert(1, 'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')
from settings import channels_standard, channels_ref, channels_le, channel_mapping_ref, channel_mapping_le
from settings import f_s, f_min, f_max

# Function to normalize the data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Read the Excel file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenge_set_path = os.path.join(desktop_path, "challenge_set.xlsx")
df = pd.read_excel(challenge_set_path)

# Initialization
all_features = []
labels = []

# Process each EDF file
for i in range(min(1, len(df))):  # Ensure we only process up to 10 files
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    subject_id = selected_row['subject_id']
    
    # Determine the appropriate channel set based on the EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    channel_names = raw.info['ch_names']
    if 'EEG FP1-REF' in channel_names:
        channels = channels_ref
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_ref)
        raw.reorder_channels(channels_standard)
    else:
        channels = channels_le
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_le)
        raw.reorder_channels(channels_standard)
    
    raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

    # Set standard montage for channel positions
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Extract data and times
    data, times = raw[:, :]

    # Select 200 seconds of data, skip the first 2 minutes (120 seconds)
    start_sample = 120 * f_s
    n_samples = 200 * f_s
    data = data[:, start_sample:start_sample + n_samples]
    times = times[start_sample:start_sample + n_samples]

    # Split the data into 50 segments, each 4 seconds long
    segment_length = 4 * f_s
    n_segments = 50

    epochs = []
    for seg_idx in range(n_segments):
        start_idx = seg_idx * segment_length
        end_idx = start_idx + segment_length
        epoch = data[:, start_idx:end_idx]
        epochs.append(epoch)

    # Convert epochs to MNE Epochs object for Autoreject
    epochs_array = np.array(epochs)
    info = mne.create_info(channels_standard, f_s, ch_types='eeg')
    epochs_mne = mne.EpochsArray(epochs_array, info)
    epochs_mne.set_montage(montage)

    # Use Autoreject to detect and repair artifacts
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs_mne)

    # Apply CAR (Common Average Referencing)
    epochs_car = epochs_clean.copy().apply_proj()
    epochs_car = epochs_car.set_eeg_reference('average', projection=False)

    # Remove mean from each epoch
    epochs_car = epochs_car.apply_function(lambda x: x - np.mean(x, axis=-1, keepdims=True))

    # Normalize the data for all channels after CAR and mean removal
    normalized_data = np.array([normalize_data(epochs_car.get_data()[:, ch].flatten()) for ch in range(epochs_car.get_data().shape[1])])
    
    # Reshape normalized data back to original epochs structure
    normalized_data = normalized_data.reshape(epochs_car.get_data().shape)

    # Calculate the average of each epoch across all segments
    mean_feature = np.mean(normalized_data, axis=0)  # Mean across all epochs

    # Flatten the mean feature to create a 1D array
    flattened_feature = mean_feature.flatten()

    # Ensure that the feature array has exactly 294 elements
    if len(flattened_feature) > 294:
        flattened_feature = flattened_feature[:294]
    elif len(flattened_feature) < 294:
        flattened_feature = np.pad(flattened_feature, (0, 294 - len(flattened_feature)), mode='constant')

    # Store the feature and the label
    all_features.append(flattened_feature)
    labels.append(subject_id)

# Combine features and labels
features_array = np.array(all_features)
labels_array = np.array(labels).reshape(-1, 1)
final_data = np.hstack((features_array, labels_array))

# Create a DataFrame for the final data
columns = [f'feature_{i+1}' for i in range(294)] + ['label']
final_df = pd.DataFrame(final_data, columns=columns)

# Create a new folder on the desktop to save the output file
output_folder = os.path.join(desktop_path, "EEG_Cleaned_Features_heldout")
os.makedirs(output_folder, exist_ok=True)

# Save the DataFrame to an Excel file
output_file = os.path.join(output_folder, "heldout_features.xlsx")
final_df.to_excel(output_file, index=False)

print("Feature extraction complete. Data saved to:", output_file)

feature_name_path = os.path.join(desktop_path, "feature_name.xlsx")
feature_name_df = pd.read_excel(feature_name_path, header=None)

# 获取feature_name.xlsx的第一列前294个字符作为特征名称
new_feature_names = feature_name_df.iloc[:294, 0].tolist()

# 确保特征名称数量为294个
if len(new_feature_names) < 294:
    new_feature_names += [f'feature_{i+1}' for i in range(len(new_feature_names), 294)]

# 将heldout_features.xlsx的特征列名替换为新的特征名称
new_columns = new_feature_names + ['Label']
final_df.columns = new_columns

# 保存修改后的DataFrame到Excel文件
output_file_modified = os.path.join(output_folder, "heldout_features.xlsx")
final_df.to_excel(output_file_modified, index=False)

print("Feature name modification complete. Modified data saved to:", output_file_modified)
