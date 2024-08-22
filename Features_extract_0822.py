import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import gc  # 导入垃圾回收模块
from settings import f_s, f_min, f_max

def extract_and_save_features(preprocessed_eeg_path, output_filename, feature_names_path, labels_df):
    # Get all .fif files in the EEG folder and sort them by the numeric part of the file name
    fif_files = [os.path.join(preprocessed_eeg_path, f) for f in os.listdir(preprocessed_eeg_path) if f.endswith('.fif')]
    fif_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

    def extract_features(epochs, f_s):
        n_epochs = epochs.get_data().shape[0]
        n_channels = epochs.get_data().shape[1]

        all_features = []

        for epoch_idx in range(n_epochs):
            epoch_features = []

            for ch_idx in range(n_channels):
                sub_segment = epochs.get_data(copy=False)[epoch_idx, ch_idx, :]

                # Time Domain 
                mean = np.mean(sub_segment)
                median = np.median(sub_segment)
                std = np.std(sub_segment)
                ptp = np.ptp(sub_segment)  # peak to peak
                mad = np.mean(np.abs(sub_segment - mean))  # Mean Absolute Deviation
                mean_square_value = np.mean(sub_segment**2)  # Mean square value
                rms = np.sqrt(np.mean(sub_segment**2))
                kurt = kurtosis(sub_segment)
                skewness = skew(sub_segment)

                # Frequency Domain 
                # Compute power spectral density (PSD)
                freqs, psd = welch(sub_segment, fs=f_s)
                
                # Compute band power in delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)
                # Integrate the PSD to get the band power

                delta_bp = np.trapz(psd[(freqs >= 1) & (freqs < 4)])
                theta_bp = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
                alpha_bp = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
                beta_bp = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
                gamma_bp = np.trapz(psd[(freqs >= 30) & (freqs < 100)])

                channel_features = ([mean, median, std, ptp, mad, mean_square_value, rms, skewness, kurt,
                                     delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp])
                epoch_features.extend(channel_features)

            all_features.append(epoch_features)

        return np.array(all_features)

    # Create a list to hold all features from all files
    all_data = []

    for i in range(1000,1200):
        if i >= len(fif_files):
            break

        fif_file = fif_files[i]
        subject_label = labels_df.iloc[i]['subject_id']
        print(f"Processing {subject_label} (file {i+1} of 200)...")

        epochs = mne.read_epochs(fif_file)
        features = extract_features(epochs, f_s)
        n_epochs = features.shape[0]

        # Add label column
        label_column = np.array([subject_label] * n_epochs).reshape(-1, 1)
        features_with_label = np.hstack((features, label_column))

        all_data.append(features_with_label)

        # Free memory
        del epochs
        del features
        gc.collect()

    # Combine all features into a single array
    final_data = np.vstack(all_data)

    # Create DataFrame
    feature_columns = [f'Feature_{i+1}' for i in range(final_data.shape[1] - 1)] + ['Label']
    df = pd.DataFrame(final_data, columns=feature_columns)

    # Save to Excel
    output_path = os.path.join(desktop_path, output_filename)
    df.to_excel(output_path, index=False)

    print(f"All features have been successfully extracted and saved to {output_path}")

    # Replace feature names
    feature_names_df = pd.read_excel(feature_names_path, header=None)
    new_feature_names = feature_names_df[0].tolist()
    df.columns = new_feature_names + ['Label']

    # Re-save
    df.to_excel(output_path, index=False)

    print(f"Feature names have been successfully replaced and the file has been saved to {output_path}")

# desktop_path 
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# path to preprocessed EEG epochs
preprocessed_eeg_val_folder_path = "D:\\Reidentification\\Epoch_val"
preprocessed_eeg_test_folder_path = "D:\\Reidentification\\Epoch_test"
preprocessed_eeg_train_folder_path = "D:\\Reidentification\\Epoch_train"
feature_names_path = os.path.join(desktop_path, 'feature_name.xlsx')

# load label file
train_labels_file_path = os.path.join(desktop_path, 'Reidentifiable_subset.xlsx')
val_labels_file_path = os.path.join(desktop_path, 'val_subset.xlsx')
test_labels_file_path = os.path.join(desktop_path, 'test_subset.xlsx')

train_labels_df = pd.read_excel(train_labels_file_path)
val_labels_df = pd.read_excel(val_labels_file_path)
test_labels_df = pd.read_excel(test_labels_file_path)

# extract training set and save features
#extract_and_save_features(preprocessed_eeg_train_folder_path, 'train_set_feature.xlsx', feature_names_path, train_labels_df)

# extract val set and save features
#extract_and_save_features(preprocessed_eeg_val_folder_path, 'val_set_feature.xlsx', feature_names_path, val_labels_df)

# extract test set and save features
extract_and_save_features(preprocessed_eeg_test_folder_path, 'test_set_feature_6.xlsx', feature_names_path, test_labels_df)
