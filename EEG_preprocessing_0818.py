import matplotlib.pyplot as plt
import os
import settings
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
all_epochs_clean = []
label = []

# Process each EDF file
for i in range(min(3, len(df))):  # Ensure we only process up to 10 files
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
    all_epochs_clean.append(mne.EpochsArray(normalized_data, info))
    
    # Subject id as label for training
    label.append(subject_id)
    
    
    
# Features extraction
def extract_features(epochs_all, f_s):
    n_files = len(epochs_all)
    n_channels = epochs_all[0].get_data(copy=False).shape[1]
    n_features = 14
    #n_epochs = epochs.shape[0]
    all_features = np.zeros((n_channels, n_features, n_files))
    
    for file_idx, epochs in enumerate(epochs_all):
        n_epochs = epochs.get_data().shape[0]
        features = np.zeros((n_channels, n_features))
        
        for ch_idx in range(n_channels):
            channel_features = []
            for epoch_idx in range(n_epochs):
                sub_segment = epochs.get_data(copy=False)[epoch_idx, ch_idx, :]
                
                # Time Domain 
                mean = np.mean(sub_segment)
                median = np.median(sub_segment)
                std = np.std(sub_segment)
                ptp = np.ptp(sub_segment)  # peak to peak 
                mad = np.mean(np.abs(sub_segment - mean))  # Mean Absolute Deviatio
                mean_square_value = np.mean(sub_segment**2)  # Mean square value
                rms = np.sqrt(np.mean(sub_segment**2))
                kurt = kurtosis(sub_segment)
                skewness = skew(sub_segment)
                
                # Frequency Domain 
                # Compute power spectral density (PSD)
                freqs, psd = welch(sub_segment, fs=f_s)
                
                # Compute band power in delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)
                delta_bp = np.trapz(psd[(freqs >= 1) & (freqs < 4)])
                theta_bp = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
                alpha_bp = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
                beta_bp = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
                gamma_bp = np.trapz(psd[(freqs >= 30) & (freqs < 100)])
                
                # Collect features from the channel
                epoch_features = ([mean, median, std, ptp, mad, mean_square_value, rms, skewness, kurt, delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp])
                channel_features.append(epoch_features)
            
            features[ch_idx] = np.mean(channel_features, axis=0)
        
        all_features[:, :, file_idx] = features
    
    return all_features 



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

features = extract_features(all_epochs_clean, f_s)