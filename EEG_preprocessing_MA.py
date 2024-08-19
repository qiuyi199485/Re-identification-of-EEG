import os
import sys
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from autoreject import AutoReject

# 插入自定义模块路径
sys.path.insert(1, 'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')
from settings import channels_standard, channels_ref, channels_le, channel_mapping_ref, channel_mapping_le
from settings import f_s, f_min, f_max

# 函数：数据归一化
def normalize_data(data):
    """将数据归一化到 [0, 1] 范围内"""
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# 函数：读取Excel文件
def load_excel_data(file_path):
    """读取指定路径的Excel文件，并返回DataFrame"""
    return pd.read_excel(file_path)

# 函数：处理单个EDF文件
def process_edf_file(row, f_s, f_min, f_max, channels_standard):
    """处理单个EDF文件，返回清理后的epochs和对应的subject_id"""
    edf_path = row['path_to_edf']
    subject_id = row['subject_id']
    
    # 读取EDF文件
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    channel_names = raw.info['ch_names']
    
    # 根据通道选择相应的设置
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
    
    # 滤波处理
    raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')
    
    # 提取数据和时间
    data, times = raw[:, :]
    
    # 选择200秒的数据，跳过前120秒
    start_sample = 120 * f_s
    n_samples = 200 * f_s
    data = data[:, start_sample:start_sample + n_samples]
    times = times[start_sample:start_sample + n_samples]
    
    # 将数据分成50个片段，每个片段4秒
    segment_length = 4 * f_s
    n_segments = 50
    epochs = [data[:, seg_idx * segment_length:(seg_idx + 1) * segment_length] for seg_idx in range(n_segments)]
    
    # 将epochs转换为MNE的Epochs对象，以便后续处理
    epochs_array = np.array(epochs)
    info = mne.create_info(channels_standard, f_s, ch_types='eeg')
    epochs_mne = mne.EpochsArray(epochs_array, info)
    
    # 返回epochs和subject_id
    return epochs_mne, subject_id

# 函数：清理和归一化epochs
def clean_and_normalize_epochs(epochs, channels_standard):
    """使用AutoReject进行伪影清理，并对数据进行归一化处理"""
    # 设置标准导联位置
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    # 使用AutoReject修正伪影
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)

    # 应用CAR（共均参考）
    epochs_car = epochs_clean.copy().apply_proj()
    epochs_car = epochs_car.set_eeg_reference('average', projection=False)

    # 去除每个epoch的均值
    epochs_car = epochs_car.apply_function(lambda x: x - np.mean(x, axis=-1, keepdims=True))

    # 对所有通道进行归一化处理
    normalized_data = np.array([normalize_data(epochs_car.get_data()[:, ch].flatten()) for ch in range(epochs_car.get_data().shape[1])])
    normalized_data = normalized_data.reshape(epochs_car.get_data().shape)

    # 返回清理和归一化后的epochs
    return mne.EpochsArray(normalized_data, epochs.info)

# 函数：保存处理后的epochs到桌面
def save_epochs(epochs, file_index):
    """将清理后的epochs保存到桌面文件"""
    # 获取桌面路径
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # 生成文件名并拼接桌面路径
    filename = f'all_epochs_clean_{file_index}.fif'
    file_path = os.path.join(desktop_path, filename)
    
    # 保存文件
    epochs.save(file_path, overwrite=True)


# 主函数：处理Excel中的所有EDF文件
def main():
    # 读取Excel文件
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
    df = load_excel_data(challenges_subset_path)
    
    all_epochs_clean = []
    label = []
    
    # 处理每个EDF文件
    for i in range(min(1, len(df))):  # 这里可以调整处理的文件数量
        selected_row = df.iloc[i]
        epochs_mne, subject_id = process_edf_file(selected_row, f_s, f_min, f_max, channels_standard)
        
        # 清理和归一化epochs
        epochs_clean = clean_and_normalize_epochs(epochs_mne, channels_standard)
        
        # 存储清理后的epochs和对应的label
        all_epochs_clean.append(epochs_clean)
        label.append(subject_id)
        
        # 保存结果
        save_epochs(epochs_clean, i)

if __name__ == "__main__":
    main()
