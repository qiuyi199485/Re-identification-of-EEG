import matplotlib.pyplot as plt
import os
import sys
import mne
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
sys.path.insert( 1,'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')
import settings

#Settings
f_s=250
f_min=1
f_max=40

# Function to normalize the data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Read the Excel file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
df = pd.read_excel(challenges_subset_path)

# Select a specific row
selected_row = df.iloc[0]  # Change the index to select a different row

# Get the path to the EDF file
edf_path = 'E:\\2024.8.11\\归档\\aaaaauqu_s001_t000.edf'

# Read the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Pick the channels 'FP1' and 'FP2'
channels = ['EEG FP1-REF']
raw.pick_channels(channels)
raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

# Extract data and times
data, times = raw[:, :]

# Select first 600 seconds of data
n_samples = f_s * 600
data = data[:, :n_samples]
times = times[:n_samples]

# Normalize the data
normalized_data = normalize_data(data[0]), normalize_data(data[1])
normalized_data_fp1 = normalize_data(data[0])

# Plot the data
plt.figure(figsize=(12, 6))

plt.plot(times, normalized_data[0], label='FP1')


plt.title('Normalized EEG Channels FP1 ')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.legend()

plt.show()

normalized_data = pd.DataFrame({
    'Time': times,
    'FP1': normalized_data_fp1,
})

export_path = os.path.join(desktop_path, "normalized_data.csv")
normalized_data.to_csv(export_path, index=False)