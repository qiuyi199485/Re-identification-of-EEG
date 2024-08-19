import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from settings import f_s, f_min, f_max
# 定义桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 假设存储了多个 .fif 文件
fif_files = [os.path.join(desktop_path, f'all_epochs_clean_{i}.fif') for i in range(2)]  # 假设有两个文件

# 读取 .fif 文件并加载数据
epochs_list = [mne.read_epochs(fif_file) for fif_file in fif_files]

# 读取保存的标签
labels = ["subject_1", "subject_2"]  # 根据实际情况调整

def extract_features(epochs, f_s):
    n_epochs = epochs.get_data().shape[0]
    n_channels = epochs.get_data().shape[1]
    n_features = 14
    
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
            delta_bp = np.trapz(psd[(freqs >= 1) & (freqs < 4)])
            theta_bp = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
            alpha_bp = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
            beta_bp = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
            gamma_bp = np.trapz(psd[(freqs >= 30) & (freqs < 100)])
            
            # Collect features from the channel
            channel_features = ([mean, median, std, ptp, mad, mean_square_value, rms, skewness, kurt, delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp])
            epoch_features.extend(channel_features)
        
        all_features.append(epoch_features)
    
    return np.array(all_features)


# 保存所有提取的特征和标签
all_data = []

for index, epochs in enumerate(epochs_list):
    subject_label = labels[index]
    print(f"Processing {subject_label}...")

    features = extract_features(epochs, f_s)
    n_epochs = features.shape[0]
    
    # 添加标签列
    label_column = np.array([subject_label] * n_epochs).reshape(-1, 1)
    features_with_label = np.hstack((features, label_column))
    
    all_data.append(features_with_label)

# 将所有数据拼接到一起
final_data = np.vstack(all_data)

# 创建 DataFrame
feature_columns = [f'Feature_{i+1}' for i in range(final_data.shape[1] - 1)] + ['Label']
df = pd.DataFrame(final_data, columns=feature_columns)

# 保存为 Excel 文件
output_path = os.path.join(desktop_path, 'extracted_features.xlsx')
df.to_excel(output_path, index=False)

print(f"Features have been successfully extracted and saved to {output_path}")
