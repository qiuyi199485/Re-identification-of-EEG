import pandas as pd
import os
import random

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 读取Excel文件
reidentifiable_file = os.path.join(desktop_path, "Reidentifiable_subset.xlsx")
val_test_file = os.path.join(desktop_path, "val_test_subset.xlsx")

reidentifiable_df = pd.read_excel(reidentifiable_file)
val_test_df = pd.read_excel(val_test_file)

# 从Reidentifiable_subset随机抽取100行
sampled_df = reidentifiable_df.sample(n=100, random_state=42)  # 设置random_state以便结果可重复

# 根据抽样的“subjec_id”筛选val_test_subset中的行
subject_ids = sampled_df["subject_id"].tolist()
filtered_df = val_test_df[val_test_df["subject_id"].isin(subject_ids)]

# 将dataframe导出为Excel文件
session_first_path = os.path.join(desktop_path, "session_first.xlsx")
session_second_path = os.path.join(desktop_path, "session_second.xlsx")

sampled_df.to_excel(session_first_path, index=False)
filtered_df.to_excel(session_second_path, index=False)

print(f"session_first.xlsx saved to {session_first_path}")
print(f"session_second.xlsx saved to {session_second_path}")

# 随机分出20行作为session_second_val，其余的作为session_second_test
session_second_val = filtered_df.sample(n=20, random_state=42)
session_second_test = filtered_df.drop(session_second_val.index)

# 将session_second_val和session_second_test分别保存为Excel文件
session_second_val_path = os.path.join(desktop_path, "session_second_val.xlsx")
session_second_test_path = os.path.join(desktop_path, "session_second_test.xlsx")

session_second_val.to_excel(session_second_val_path, index=False)
session_second_test.to_excel(session_second_test_path, index=False)

print(f"session_second_val.xlsx saved to {session_second_val_path}")
print(f"session_second_test.xlsx saved to {session_second_test_path}")

