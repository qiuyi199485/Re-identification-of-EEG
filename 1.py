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

# 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
sys.path.insert(1, 'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')

from settings import channels_standard, channels_ref, channels_le, channel_mapping_ref, channel_mapping_le
from settings import f_s, f_min, f_max

# 数据归一化函数
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# 读取Excel文件
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
df = pd.read_excel(challenges_subset_path)

# 存储所有EEG数据
all_epochs_clean = []

# 处理每个EDF文件
for i in range(min(3, len(df))):  # 确保最多处理3个文件
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    
    # 根据EDF文件确定适当的通道集
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

    # 设置标准导联位置
    montage = mne.channels.make_standard_montage('standard_1020')
    # raw.set_montage(montage)
    
    # 提取数据和时间
    data, times = raw[:, :]

    # 选择200秒的数据，跳过前2分钟（120秒）
    start_sample = 120 * f_s
    n_samples = 200 * f_s
    data = data[:, start_sample:start_sample + n_samples]
    times = times[start_sample:start_sample + n_samples]

    # 归一化所有通道的数据
    normalized_data = np.array([normalize_data(data[ch]) for ch in range(data.shape[0])])
    
    # 将数据分成50段，每段4秒
    segment_length = 4 * f_s
    n_segments = 50

    epochs = []
    for seg_idx in range(n_segments):
        start_idx = seg_idx * segment_length
        end_idx = start_idx + segment_length
        epoch = normalized_data[:, start_idx:end_idx]
        epochs.append(epoch)

    # 将段数据转换为MNE Epochs对象以进行自动伪影检测和修复
    epochs_array = np.array(epochs)
    info = mne.create_info(channels_standard, f_s, ch_types='eeg')
    epochs_mne = mne.EpochsArray(epochs_array, info)
    epochs_mne.set_montage(montage)

    # 使用AutoReject检测和修复伪影
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs_mne)
    
    # 存储清理后的EEG数据
    all_epochs_clean.append(epochs_clean)

# 特征提取函数
def extract_features(epochs_list, f_s):
    n_files = len(epochs_list)
    n_channels = epochs_list[0].get_data().shape[1]
    n_features = 11  # 每个信号提取的特征数
    
    # 初始化特征矩阵，大小为 (n_channels, n_features, n_files)
    all_features = np.zeros((n_channels, n_features, n_files))
    
    for file_idx, epochs in enumerate(epochs_list):
        n_epochs = epochs.get_data().shape[0]
        features = np.zeros((n_channels, n_features))
        
        for ch_idx in range(n_channels):
            channel_features = []
            for epoch_idx in range(n_epochs):
                sub_segment = epochs.get_data()[epoch_idx, ch_idx, :]
                
                # 时域特征
                mean = np.mean(sub_segment)
                median = np.median(sub_segment)
                std = np.std(sub_segment)
                rms = np.sqrt(np.mean(sub_segment**2))
                kurt = kurtosis(sub_segment)
                skewness = skew(sub_segment)
                
                # 频域特征
                # 计算功率谱密度（PSD）
                freqs, psd = welch(sub_segment, fs=f_s)
                
                # 计算各频段的功率：delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)
                delta_bp = np.trapz(psd[(freqs >= 1) & (freqs < 4)])
                theta_bp = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
                alpha_bp = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
                beta_bp = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
                gamma_bp = np.trapz(psd[(freqs >= 30) & (freqs < 100)])
                
                # 收集通道特征
                epoch_features = ([mean, median, std, rms, skewness, kurt, delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp])
                channel_features.append(epoch_features)
            
            features[ch_idx] = np.mean(channel_features, axis=0)
        
        all_features[:, :, file_idx] = features
    
    return all_features

# 绘制清理后的EEG信号
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

# 提取特征
features = extract_features(all_epochs_clean, f_s)
print(features.shape)  # 输出特征矩阵的形状，应为 (21, 11, 3)
