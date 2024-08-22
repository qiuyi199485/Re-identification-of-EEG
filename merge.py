import os
import pandas as pd

# 获取桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 定义文件夹路径
folder_path = os.path.join(desktop_path, "val")

# 获取文件夹中所有xlsx文件的文件名（按顺序）
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])

# 初始化一个空的DataFrame用于存储合并后的数据
combined_df = pd.DataFrame()

# 逐个读取文件并合并
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    
    # 读取当前xlsx文件
    df = pd.read_excel(file_path)
    
    if combined_df.empty:
        # 如果是第一个文件，直接将其加入combined_df
        combined_df = df
    else:
        # 如果不是第一个文件，去掉当前文件的标题行（第一行）后，合并到combined_df
        combined_df = pd.concat([combined_df, df.iloc[1:]], ignore_index=True)

# 导出合并后的数据到新的xlsx文件
output_path = os.path.join(desktop_path, "val_set_features.xlsx")
combined_df.to_excel(output_path, index=False)

print(f"文件已成功导出到: {output_path}")
