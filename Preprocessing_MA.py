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
channels_standard = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'A1', 'T3', 'C3',
                     'Cz', 'C4', 'T4', 'A2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

channels_ref = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
                'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
                'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF']

channels_le = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F7-LE', 'EEG F3-LE', 'EEG FZ-LE', 'EEG F4-LE', 'EEG F8-LE',
               'EEG A1-LE', 'EEG T3-LE', 'EEG C3-LE', 'EEG CZ-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG A2-LE',
               'EEG T5-LE', 'EEG P3-LE', 'EEG PZ-LE', 'EEG P4-LE', 'EEG T6-LE', 'EEG O1-LE', 'EEG O2-LE']

# Create and set montage


# Mapping to standard 10-20 channel names
channel_mapping_ref = {
    'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4',
    'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4',
    'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T7', 'EEG T4-REF': 'T8', 'EEG T5-REF': 'P7', 'EEG T6-REF': 'P8',
    'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', 'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz',
    'EEG PZ-REF': 'Pz'}

channel_mapping_le = {
    'EEG FP1-LE': 'Fp1', 'EEG FP2-LE': 'Fp2', 'EEG F3-LE': 'F3',
    'EEG F4-LE': 'F4', 'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3',
    'EEG P4-LE': 'P4', 'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7',
    'EEG F8-LE': 'F8', 'EEG T3-LE': 'T7', 'EEG T4-LE': 'T8', 'EEG T5-LE': 'P7',
    'EEG T6-LE': 'P8', 'EEG A1-LE': 'A1', 'EEG A2-LE': 'A2', 'EEG FZ-LE': 'Fz',
    'EEG CZ-LE': 'Cz', 'EEG PZ-LE': 'Pz'
}


# Load the challenges dataset
def load_excel_file(file_path):
    return pd.read_excel(file_path)


# Bandpass filter and channel rename
def process_edf_file(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    channel_names = raw.info['ch_names']
    
    if 'EEG FP1-REF' in channel_names:
        channels = channels_ref
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_ref)
    else:
        channels = channels_le
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_le)
    
    raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw


# skip start 2 mins (120 sec) and get 200 sec signals， then normalize the signals.
def extract_and_normalize_data(raw, start_time=120, duration=200, f_s=f_s):
    start_sample = start_time * f_s
    n_samples = duration * f_s
    data, times = raw[:, :]
    data = data[:, start_sample:start_sample + n_samples]
    times = times[start_sample:start_sample + n_samples]
    normalized_data = np.array([normalize_data(data[ch]) for ch in range(data.shape[0])])
    return normalized_data, times


# 50 Segments (each 4 sec) for Autoreject Algorithmuns
def segment_data(normalized_data, segment_length=4*f_s, n_segments=50):
    epochs = []
    for seg_idx in range(n_segments):
        start_idx = seg_idx * segment_length
        end_idx = start_idx + segment_length
        epoch = normalized_data[:, start_idx:end_idx]
        epochs.append(epoch)
    return np.array(epochs)



# Autoreject to clean EEG signal
def apply_autoreject(epochs_array, channels_standard, montage):
    info = mne.create_info(channels_standard, f_s, ch_types='eeg')
    epochs_mne = mne.EpochsArray(epochs_array, info)
    epochs_mne.set_montage(montage)
    
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs_mne)
    return epochs_clean




