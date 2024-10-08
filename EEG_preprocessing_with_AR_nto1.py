import matplotlib.pyplot as plt
import os
import sys
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from autoreject import AutoReject
from scipy.signal import welch
from scipy.stats import kurtosis, skew

sys.path.insert(1, 'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')
from settings import channels_standard, channels_ref, channels_le, channel_mapping_ref, channel_mapping_le
from settings import f_s, f_min, f_max

# Function to normalize the data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Read the Excel file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
df = pd.read_excel(challenges_subset_path)

# Initialization
all_epochs_data = []
all_epochs_labels = []

# Process each EDF file
for i in range(min(2, len(df))):  # Ensure we only process up to 20 files
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

    # Store the cleaned and normalized epochs
    all_epochs_data.append(normalized_data)
    
    # Subject id as label for training
    all_epochs_labels.extend([subject_id] * normalized_data.shape[0])

# Combine all epochs into one large Epochs object
combined_epochs_data = np.concatenate(all_epochs_data, axis=0)
combined_epochs_info = mne.create_info(channels_standard, f_s, ch_types='eeg')
combined_epochs = mne.EpochsArray(combined_epochs_data, combined_epochs_info)

# Save the combined epochs to a single FIF file
combined_filename = os.path.join(desktop_path, 'combined_all_epochs_clean.fif')
combined_epochs.save(combined_filename, overwrite=True)
