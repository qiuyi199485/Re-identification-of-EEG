import os
import mne
import matplotlib.pyplot as plt
import pandas as pd
from settings import f_s, f_min, f_max
# 定义桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 假设存储了多个 .fif 文件
fif_files = [os.path.join(desktop_path, f'all_epochs_clean_{i}.fif') for i in range(2)]  # 假设有两个文件

# 读取 .fif 文件并加载数据
epochs_list = [mne.read_epochs(fif_file) for fif_file in fif_files]

# 读取保存的标签
labels = ["subject_1", "subject_2"]  # 根据实际情况调整

# 选择要查看的 index
index = 1

# 获取对应的 epoch 数据
epoch_data = epochs_list[index]

# 获取对应的标签
subject_label = labels[index]
print(f"Subject label: {subject_label}")

# 绘制第一个 epoch
# 绘制第一个 epoch
epoch_to_plot = epoch_data[0]  # 获取第一个 epoch
epoch_to_plot.plot(scalings='auto')  # 使用自动比例进行绘图

plt.show()


