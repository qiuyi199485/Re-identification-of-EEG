import pandas as pd
import os

# 定义特征列表
features = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'A1', 'T3', 'C3',
            'Cz', 'C4', 'T4', 'A2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

# 定义统计量列表
statistics = ['mean', 'median', 'std', 'ptp', 'mad', 'msv', 'rms', 
              'skewness', 'kurt', 'delta_bp', 'theta_bp', 'alpha_bp', 'beta_bp', 'gamma_bp']

# 生成组合后的特征列表
combined_features = [f"{feature}_{stat}" for feature in features for stat in statistics]

# 将组合后的特征列表转换为DataFrame，并不添加列标题
df = pd.DataFrame(combined_features)

# 获取桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 定义文件路径
file_path = os.path.join(desktop_path, 'feature_name_expanded.xlsx')

# 将DataFrame保存为Excel文件，不保存列名
df.to_excel(file_path, index=False, header=False)

print(f"文件已保存到 {file_path}")
