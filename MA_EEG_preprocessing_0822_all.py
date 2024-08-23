import os
import sys
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, 'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')
from settings import channels_standard, channels_ref, channels_le, channel_mapping_ref, channel_mapping_le
from settings import f_s, f_min, f_max

# Function to normalize the data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

def process_data(df, output_folder):
    # Initialization

    # Process each EDF file
    for i in range(1701,len(df)):  
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
        n_segments = 5

        epochs = []
        for seg_idx in range(n_segments):
            start_idx = seg_idx * segment_length
            end_idx = start_idx + segment_length
            epoch = data[:, start_idx:end_idx]
            epochs.append(epoch)

        # Convert epochs to MNE Epochs object
        epochs_array = np.array(epochs)
        info = mne.create_info(channels_standard, f_s, ch_types='eeg')
        epochs_mne = mne.EpochsArray(epochs_array, info)
        epochs_mne.set_montage(montage)

        # Apply CAR (Common Average Referencing)
        epochs_car = epochs_mne.copy().apply_proj()
        epochs_car = epochs_car.set_eeg_reference('average', projection=False)

        # Remove mean from each epoch
        epochs_car = epochs_car.apply_function(lambda x: x - np.mean(x, axis=-1, keepdims=True))

        # Normalize the data for all channels after CAR and mean removal
        normalized_data = np.array([normalize_data(epochs_car.get_data()[:, ch].flatten()) for ch in range(epochs_car.get_data().shape[1])])

        # Reshape normalized data back to original epochs structure
        normalized_data = normalized_data.reshape(epochs_car.get_data().shape)

        # Store the cleaned and normalized epochs
        cleaned_epochs = mne.EpochsArray(normalized_data, info)

        # Create the folder structure on the output path
        os.makedirs(output_folder, exist_ok=True)
        
        # Define the filename based on the subject ID
        subset_type = os.path.basename(output_folder)
        filename = os.path.join(output_folder, f'subject_{i+1}_{subset_type}.fif')

        # Save the cleaned epochs to the new folder
        cleaned_epochs.save(filename, overwrite=True)

        print(f"Processed and saved {filename}")

# Paths for the new Excel files
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
val_subset_path = os.path.join(desktop_path, "val_subset.xlsx")
test_subset_path = os.path.join(desktop_path, "test_subset.xlsx")
train_subset_path = os.path.join(desktop_path, "Reidentifiable_subset.xlsx")

# Read the Excel files
df_train = pd.read_excel(train_subset_path)
df_val = pd.read_excel(val_subset_path)
df_test = pd.read_excel(test_subset_path)

# Process and save the training subset (can change the path to desktop)
#train_epoch_path = os.path.join("D:\\Reidentification", "Epoch_train")
#process_data(df_train, "D:\\Reidentification\\Epoch_train")
#print(f"The preprocessed epochs of training set have been saved to {train_epoch_path}")

# Process and save the validation subset
#val_epoch_path = os.path.join("D:\\Reidentification", "Epoch_val")
#process_data(df_val, val_epoch_path)
#print(f"The preprocessed epochs of validation set have been saved to {val_epoch_path}")

# Process and save the test subset
test_epoch_path = os.path.join("D:\\Reidentification", "Epoch_test")
process_data(df_test, test_epoch_path)
print(f"The preprocessed epochs of test set have been saved to {test_epoch_path}")
