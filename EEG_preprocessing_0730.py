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

# Mapping to standard 10-20 channel names
channel_mapping_ref = {
    'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4',
    'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4',
    'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5', 'EEG T6-REF': 'T6',
    'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', 'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz',
    'EEG PZ-REF': 'Pz'}

channel_mapping_le = {
    'EEG FP1-LE': 'Fp1', 'EEG FP2-LE': 'Fp2', 'EEG F3-LE': 'F3',
    'EEG F4-LE': 'F4', 'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3',
    'EEG P4-LE': 'P4', 'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7',
    'EEG F8-LE': 'F8', 'EEG T3-LE': 'T3', 'EEG T4-LE': 'T4', 'EEG T5-LE': 'T5',
    'EEG T6-LE': 'T6', 'EEG A1-LE': 'A1', 'EEG A2-LE': 'A2', 'EEG FZ-LE': 'Fz',
    'EEG CZ-LE': 'Cz', 'EEG PZ-LE': 'Pz'
}
# Read the Excel file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
df = pd.read_excel(challenges_subset_path)

# Process each EDF file
for i in range(min(1, len(df))):  # Ensure we only process up to 10 files
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    
    # Determine the appropriate channel set based on the EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    channel_names = raw.info['ch_names']
    if 'EEG FP1-REF' in channel_names:
        channels = channels_ref
        
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_ref)
        raw.reorder_channels(channels_standard)
        print(raw.ch_names)
    
    else:
        channels = channels_le
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_le)
        raw.reorder_channels(channels_standard)
        print(raw.ch_names)

    
    
    raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

    # Set standard montage for channel positions
    montage = mne.channels.make_standard_montage('standard_1020')
    #raw.set_montage(montage)
    
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
    info = mne.create_info(channels_standard, f_s, ch_types='eeg')
    epochs_mne = mne.EpochsArray(epochs_array, info)
    epochs_mne.set_montage(montage)

    # Use Autoreject to detect and repair artifacts
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs_mne)

    # Plot the cleaned data for the first epoch as an example
    #plt.figure(figsize=(15, 10))
    #for ch in range(epochs_clean.get_data().shape[1]):
       # plt.plot(np.linspace(0, 4, segment_length), epochs_clean.get_data()[0, ch], label=info['ch_names'][ch])
    #plt.title(f'Cleaned EEG Channels for File {i+1} - Segment 1')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Normalized Amplitude')
    #plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    #plt.show()
    
    
    
# 特征提取函数
def extract_features(epochs, f_s):
    features = []
    for epoch in epochs:
        epoch_features = []
        for sub_segment in epoch:
            
            # Time Domain 
            mean = np.mean(sub_segment)
            median = np.median(sub_segment)
            std = np.std(sub_segment)
            rms = np.sqrt(np.mean(sub_segment**2))
            kurt = kurtosis(sub_segment)
            skewness = skew(sub_segment)
            
            
            # Frequency Domain 
            # Compute power spectral density (PSD)
            freqs, psd = welch(sub_segment, fs=f_s)
            
            # Compute band power in delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz) , and gama (30-100 Hz) bands
            delta_bp = np.trapz(psd[(freqs >= 1) & (freqs < 4)])
            theta_bp = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
            alpha_bp = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
            beta_bp = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
            gamma_bp = np.trapz(psd[(freqs >= 30) & (freqs < 100)])
            
            # Collect features for the channel
            epoch_features.extend([mean, median, std, rms, skewness, kurt, delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp])
        features.append(epoch_features)
    return np.array(features)



#Plot cleaned EEG signal
def plot_cleaned_eeg(epochs_clean):
    n_channels, n_times = epochs_clean.get_data().shape[1:3]
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)

    for ch_idx in range(n_channels):
        ax = axes[ch_idx] if n_channels > 1 else axes
        ch_name = epochs_clean.ch_names[ch_idx]
        data = epochs_clean.get_data()[:, ch_idx, :].flatten()
        times = np.linspace(0, len(data) / f_s, len(data))
        
        ax.plot(times, data, label=f'Channel {ch_name}')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


#plot_cleaned_eeg(epochs_clean)


features = extract_features(epochs_clean.get_data(), f_s)