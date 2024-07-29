import matplotlib.pyplot as plt
import os
import sys
import mne
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import pickle


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

# Function to fit AR model
def fit_ar_model(signal, lags):
    model = AutoReg(signal, lags=lags)
    model_fit = model.fit()
    return model_fit

# Channel sets
channels_ref = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
                'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
                'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF']

channels_le = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F7-LE', 'EEG F3-LE', 'EEG FZ-LE', 'EEG F4-LE', 'EEG F8-LE',
               'EEG A1-LE', 'EEG T3-LE', 'EEG C3-LE', 'EEG CZ-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG A2-LE',
               'EEG T5-LE', 'EEG P3-LE', 'EEG PZ-LE', 'EEG P4-LE', 'EEG T6-LE', 'EEG O1-LE', 'EEG O2-LE']

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
    raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

    # Extract data and times
    data, times = raw[:, :]

    # Select first 600 seconds of data
    n_samples = f_s * 600
    data = data[:, :n_samples]
    times = times[:n_samples]

    # Normalize the data for all channels
    normalized_data = np.array([normalize_data(data[ch]) for ch in range(data.shape[0])])

    # Fit AR model for all channels
    lags = 6  # Number of lags for AR model, can be tuned
    ar_models = [fit_ar_model(normalized_data[ch], lags) for ch in range(normalized_data.shape[0])]

    

    # Plot the data for all channels
    #plt.figure(figsize=(15, 10))
    #for ch in range(normalized_data.shape[0]):
        #plt.plot(times, normalized_data[ch], label=channels[ch])
    #plt.title(f'Normalized EEG Channels for File {i+1}')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Normalized Amplitude')
    #plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
   # plt.show()

    # Save normalized data
    #normalized_data_df = pd.DataFrame(normalized_data.T, columns=channels)
    #normalized_data_df['Time'] = times
    #export_path = os.path.join(desktop_path, f"normalized_data_{i+1}.csv")
    #normalized_data_df.to_csv(export_path, index=False)
    
# Print AR model parameters for all channels
print(f"\nEDF File: {edf_path}")
for idx, ar_model in enumerate(ar_models):
    print(f"Channel {channels[idx]} AR-params：", ar_model.params)



def save_ar_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        
            
for idx, ar_model in enumerate(ar_models):
    model_filename = os.path.join(desktop_path, f"ar_model_file_{i+1}_channel_{channels[idx]}.pkl")
    save_ar_model(ar_model, model_filename)
    print(f"Model for channel {channels[idx]} saved as {model_filename}")