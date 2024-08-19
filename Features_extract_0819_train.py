import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from settings import f_s, f_min, f_max

# 定义桌面路径并指定EEG文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
eeg_folder_path = os.path.join(desktop_path, 'EEG_Cleaned_Epochs_train')

# 读取标签文件
labels_file_path = os.path.join(desktop_path, 'challenges_subset.xlsx')
labels_df = pd.read_excel(labels_file_path)

# 获取 subject_id 列的值作为标签
labels = labels_df['subject_id'].tolist()

# 获取EEG文件夹中的所有 .fif 文件，并按照文件名中的数字部分排序
fif_files = [os.path.join(eeg_folder_path, f) for f in os.listdir(eeg_folder_path) if f.endswith('.fif')]
fif_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

# 检查文件数量和标签数量是否匹配
if len(fif_files) != len(labels):
    raise ValueError("文件数量与标签数量不匹配，请检查输入数据。")

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

for index, fif_file in enumerate(fif_files):
    subject_label = labels[index]
    print(f"Processing {subject_label}...")

    # 逐个加载 .fif 文件并提取特征
    epochs = mne.read_epochs(fif_file)
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


extracted_features_path = os.path.join(desktop_path, 'extracted_features.xlsx')
feature_names_path = os.path.join(desktop_path, 'feature_name.xlsx')

# 读取数据
extracted_features_df = pd.read_excel(extracted_features_path)
feature_names_df = pd.read_excel(feature_names_path, header=None)

# 使用 feature_name.xlsx 的第一列替换 extracted_features.xlsx 的前294列的列名
new_feature_names = feature_names_df[0].tolist()
extracted_features_df.columns = new_feature_names + extracted_features_df.columns[294:].tolist()

# 保存替换列名后的数据，覆盖原始文件
extracted_features_df.to_excel(extracted_features_path, index=False)

print(f"Feature names have been successfully replaced and the file has been saved to {extracted_features_path}")