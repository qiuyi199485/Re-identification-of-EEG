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

for i in range(min(2, len(df))):  # Ensure we only process up to 10 files
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    
    # Determine the appropriate channel set based on the EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    channel_names = raw.info['ch_names']
    if 'EEG FP1-REF' in channel_names:
        channels = channels_ref
        
        raw.pick_channels(channels)   # keep channel
        raw.rename_channels(channel_mapping_ref)
        raw.reorder_channels(channels_standard)
        print(raw.ch_names)
    
    else:
        channels = channels_le
        raw.pick_channels(channels)
        raw.rename_channels(channel_mapping_le)
        raw.reorder_channels(channels_standard)
        print(raw.ch_names)
        
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    print(f"Montage channels for file {i+1}: {montage.ch_names}")    