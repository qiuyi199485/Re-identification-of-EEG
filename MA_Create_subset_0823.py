import pandas as pd
import os
import random


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 读取Excel文件
reidentifiable_file = os.path.join(desktop_path, "Reidentifiable_subset.xlsx")
val_test_file = os.path.join(desktop_path, "val_test_subset.xlsx")

reidentifiable_df = pd.read_excel(reidentifiable_file)
val_test_df = pd.read_excel(val_test_file)

# 从Reidentifiable_subset随机抽取10行
sampled_df = reidentifiable_df.sample(n=10, random_state=42)  # 设置random_state以便结果可重复

# 根据抽样的“subjec_id”筛选val_test_subset中的行
subject_ids = sampled_df["subject_id"].tolist()
filtered_df = val_test_df[val_test_df["subject_id"].isin(subject_ids)]

# 将dataframe导出为Excel文件
session_1_path = os.path.join(desktop_path, "Session_1.xlsx")
session_2_path = os.path.join(desktop_path, "Session_2.xlsx")

sampled_df.to_excel(session_1_path, index=False)
filtered_df.to_excel(session_2_path, index=False)

print(f"Session_1.xlsx saved to {session_1_path}")
print(f"Session_2.xlsx saved to {session_2_path}")
