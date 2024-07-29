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
from sklearn.metrics import mean_squared_error

# Function to normalize the data
def normalize_data(data, scaler):
    return scaler.transform(data.reshape(-1, 1)).flatten()

# 读取测试数据文件路径
test_subset_path = os.path.join(desktop_path, "test_subset.xlsx")
test_df = pd.read_excel(test_subset_path)

# 用于存储模型评估结果
evaluation_results = []

# 处理每个测试EDF文件
for i in range(min(10, len(test_df))):  # 确保只处理最多10个文件
    selected_row = test_df.iloc[i]
    test_edf_path = selected_row['path_to_edf']
    
    # 确定EDF文件的通道类型
    test_raw = mne.io.read_raw_edf(test_edf_path, preload=True)
    test_channel_names = test_raw.info['ch_names']
    if 'EEG FP1-REF' in test_channel_names:
        test_channels = channels_ref
    else:
        test_channels = channels_le

    test_raw.pick_channels(test_channels)
    test_raw.filter(f_min, f_max, fir_design='firwin', skip_by_annotation='edge')

    # 提取数据和时间
    test_data, test_times = test_raw[:, :]

    # 选择前600秒的数据
    test_n_samples = f_s * 600
    test_data = test_data[:, :test_n_samples]
    test_times = test_times[:test_n_samples]

    # 归一化测试数据
    scaler = MinMaxScaler()
    scaler.fit(test_data.reshape(-1, 1))
    normalized_test_data = np.array([normalize_data(test_data[ch], scaler) for ch in range(test_data.shape[0])])

    # 评估每个通道的AR模型
    for ch in range(normalized_test_data.shape[0]):
        # 拟合AR模型
        model = fit_ar_model(normalized_test_data[ch], lags)
        # 预测
        predictions = model.predict(start=lags, end=len(normalized_test_data[ch])-1)
        # 计算均方误差
        mse = mean_squared_error(normalized_test_data[ch][lags:], predictions)
        
        # 存储结果
        evaluation_results.append({
            'File': test_edf_path,
            'Channel': test_channels[ch],
            'MSE': mse
        })

# 转换为DataFrame并显示结果
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df