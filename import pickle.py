import matplotlib.pyplot as plt
import os
import sys
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from autoreject import AutoReject

sys.path.insert(1, 'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')
import settings

# Settings
f_s = 250
f_min = 1
f_max = 40

# Function to normalize the data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Channel sets
channels_ref = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
                'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
                'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF']

channels_le = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F7-LE', 'EEG F3-LE', 'EEG FZ-LE', 'EEG F4-LE', 'EEG F8-LE',
               'EEG A1-LE', 'EEG T3-LE', 'EEG C3-LE', 'EEG CZ-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG A2-LE',
               'EEG T5-LE', 'EEG P3-LE', 'EEG PZ-LE', 'EEG P4-LE', 'EEG T6-LE', 'EEG O1-LE', 'EEG O2-LE']

# Mapping to standard 10-20 channel names
channel_mapping = {
    'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4',
    'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4',
    'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T7', 'EEG T4-REF': 'T8', 'EEG T5-REF': 'P7', 'EEG T6-REF': 'P8',
    'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', 'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz',
    'EEG PZ-REF': 'Pz', 'EEG FP1-LE': 'Fp1', 'EEG FP2-LE': 'Fp2', 'EEG F3-LE': 'F3',
    'EEG F4-LE': 'F4', 'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3',
    'EEG P4-LE': 'P4', 'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7',
    'EEG F8-LE': 'F8', 'EEG T3-LE': 'T7', 'EEG T4-LE': 'T8', 'EEG T5-LE': 'P7',
    'EEG T6-LE': 'P8', 'EEG A1-LE': 'A1', 'EEG A2-LE': 'A2', 'EEG FZ-LE': 'Fz',
    'EEG CZ-LE': 'Cz', 'EEG PZ-LE': 'Pz'
}

# Read the Excel file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
df = pd.read_excel(challenges_subset_path)

# Process each EDF file
for i in range(min(10, len(df))):  # Ensure we only process up to 10 files
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    
    # Determine the appropriate channel set based on the EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    channel_names = raw.info['ch_names']
    if 'EEG FP1-REF' in channel_names:
        channels = channels_ref
    else:
        channels = channels_le

    raw.pick_channels(channels)
    raw.rename_channels(channel_mapping)
    raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

    # Set standard montage for channel positions
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Extract data and times
    data, times = raw[:, :]

    # Select 200 seconds of data, skip the first 2 minutes (120 seconds)
    start_sample = 120 * f_s
    n_samples = 200 * f_s
    data = data[:, start_sample:start_sample + n_samples]
    times = times[start_sample:start_sample + n_samples]

    # Normalize the data for all channels
    normalized_data = np.array([normalize_data(data[ch]) for ch in range(data.shape[0])])

    # Split the data into 50 segments, each 4 seconds long
    segment_length = 4 * f_s
    n_segments = 50

    epochs = []
    for seg_idx in range(n_segments):
        start_idx = seg_idx * segment_length
        end_idx = start_idx + segment_length
        epoch = normalized_data[:, start_idx:end_idx]
        epochs.append(epoch)

    # Convert epochs to MNE Epochs object for Autoreject
    epochs_array = np.array(epochs)
    info = mne.create_info([channel_mapping[ch] for ch in channels], f_s, ch_types='eeg')
    epochs_mne = mne.EpochsArray(epochs_array, info)
    epochs_mne.set_montage(montage)

    # Use Autoreject to detect and repair artifacts
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs_mne)

    # Plot the cleaned data for the first epoch as an example
    plt.figure(figsize=(15, 10))
    for ch in range(epochs_clean.get_data().shape[1]):
        plt.plot(np.linspace(0, 4, segment_length), epochs_clean.get_data()[0, ch], label=info['ch_names'][ch])
    plt.title(f'Cleaned EEG Channels for File {i+1} - Segment 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()

    # Save cleaned epochs if needed
    # epochs_clean.save(f'cleaned_epochs_file_{i+1}-epo.fif', overwrite=True)
