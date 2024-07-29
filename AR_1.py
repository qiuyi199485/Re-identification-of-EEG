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

# Function to extract features
def extract_features(eeg_data, fs):
    num_samples, num_channels = eeg_data.shape
    features = []

    for ch in range(num_channels):
        signal = eeg_data[:, ch]
        
        # 时域特征
        mean = np.mean(signal)
        std_dev = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        kurt = kurtosis(signal)
        skewness = skew(signal)
        
        # 频域特征（使用Welch方法计算功率谱密度）
        freqs, psd = welch(signal, fs=fs, nperseg=128)
        
        # 提取各频带的能量
        delta_band = np.trapz(psd[(freqs >= 0.5) & (freqs < 4)])
        theta_band = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
        alpha_band = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
        beta_band = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
        gamma_band = np.trapz(psd[(freqs >= 30) & (freqs < 100)])
        
        # 将所有特征组合在一起
        channel_features = [mean, std_dev, rms, kurt, skewness,
                            delta_band, theta_band, alpha_band, beta_band, gamma_band]
        features.append(channel_features)

    return np.array(features)

# Function to fit AR model using enhanced features
def fit_ar_model(signal, features, lags):
    # Combine the signal with the features to create an enhanced feature matrix
    enhanced_features = np.column_stack([signal, features])
    # Fit the AR model
    model = AutoReg(enhanced_features, lags=lags)
    model_fit = model.fit()
    return model_fit

# Read the Excel file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
df = pd.read_excel(challenges_subset_path)

# Select a specific row
selected_row = df.iloc[0]  # Change the index to select a different row

# Get the path to the EDF file
edf_path = selected_row['path_to_edf']

# Read the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Pick the channels 'FP1' and 'FP2'
channels = ['EEG FP1-REF', 'EEG FP2-REF']
raw.pick_channels(channels)
raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

# Extract data and times
data, times = raw[:, :]

# Select first 600 seconds of data
n_samples = f_s * 600
data = data[:, :n_samples]
times = times[:n_samples]

# Normalize the data
normalized_data_fp1 = normalize_data(data[0])
normalized_data_fp2 = normalize_data(data[1])

# Extract features
features = extract_features(data.T, f_s)
print("提取的特征：", features)

# Fit AR model using enhanced features
lags = 6  # Number of lags for AR model, can be tuned
ar_model_fp1 = fit_ar_model(normalized_data_fp1, features[0], lags)
ar_model_fp2 = fit_ar_model(normalized_data_fp2, features[1], lags)

print("FP1 AR模型参数：", ar_model_fp1.params)
print("FP2 AR模型参数：", ar_model_fp2.params)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(times, normalized_data_fp1, label='FP1')
plt.plot(times, normalized_data_fp2, label='FP2')
plt.title('Normalized EEG Channels FP1 and FP2')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.show()

# Save normalized data
normalized_data_df = pd.DataFrame({
    'Time': times,
    'FP1': normalized_data_fp1,
    'FP2': normalized_data_fp2,
})
export_path = os.path.join(desktop_path, "normalized_data.csv")
normalized_data_df.to_csv(export_path, index=False)

# Save extracted features
features_df = pd.DataFrame(features, columns=[
    'mean', 'std_dev', 'rms', 'kurt', 'skewness',
    'delta_band', 'theta_band', 'alpha_band', 'beta_band', 'gamma_band'
])
features_export_path = os.path.join(desktop_path, "extracted_features.csv")
features_df.to_csv(features_export_path, index=False)