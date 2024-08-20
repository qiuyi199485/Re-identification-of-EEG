import os
import pandas as pd

# 使用 os.path.join 来构建文件路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 第一个文件: dataframe.xlsx
file_name_1 = "dataframe.xlsx"
file_path_1 = os.path.join(desktop_path, file_name_1)

# 读取 Excel 文件
df1 = pd.read_excel(file_path_1)

# 对每个 subject_id 只取一次 sex 的值（假设每个 subject_id 对应的 sex 唯一）
unique_subjects_1 = df1.drop_duplicates(subset='subject_id')

# 统计整体 sex 中 'm' 和 'f' 的总数量
sex_counts_1 = unique_subjects_1['sex'].value_counts()

# 统计 subject_class 为 "D" 的 subject 的 sex 数量
subject_class_d = unique_subjects_1[unique_subjects_1['subject_class'] == 'D']
sex_counts_d = subject_class_d['sex'].value_counts()

# 计算整体和 subject_class 为 "D" 的 m/f ratio
m_count_1 = sex_counts_1.get('m', 0)
f_count_1 = sex_counts_1.get('f', 0)
m_f_ratio_1 = m_count_1 / f_count_1 if f_count_1 != 0 else float('inf')

m_count_d = sex_counts_d.get('m', 0)
f_count_d = sex_counts_d.get('f', 0)
m_f_ratio_d = m_count_d / f_count_d if f_count_d != 0 else float('inf')

# 第二个文件: Reidentifiable_subset.xlsx
file_name_2 = "Reidentifiable_subset.xlsx"
file_path_2 = os.path.join(desktop_path, file_name_2)

# 读取 Excel 文件
df2 = pd.read_excel(file_path_2)

# 对每个 subject_id 只取一次 sex 的值
unique_subjects_2 = df2.drop_duplicates(subset='subject_id')

# 统计 sex 中 'm' 和 'f' 的总数量
sex_counts_2 = unique_subjects_2['sex'].value_counts()

# 计算 m/f ratio
m_count_2 = sex_counts_2.get('m', 0)
f_count_2 = sex_counts_2.get('f', 0)
m_f_ratio_2 = m_count_2 / f_count_2 if f_count_2 != 0 else float('inf')

# 第三个文件: val_set_feature.xlsx
file_name_3 = "val_set_feature.xlsx"
file_path_3 = os.path.join(desktop_path, file_name_3)

# 读取 Excel 文件并提取 Label 列的唯一值
df3 = pd.read_excel(file_path_3)
unique_labels = df3['Label'].unique()

# 使用 Label 去 Reidentifiable_subset.xlsx 文件中查找对应的 subject_id 和 sex
matching_subjects = df2[df2['subject_id'].isin(unique_labels)]

# 统计这些匹配的 subject_id 的 sex 数量
sex_counts_labels = matching_subjects['sex'].value_counts()

# 计算这些 Label 所对应的 m/f ratio
m_count_labels = sex_counts_labels.get('m', 0)
f_count_labels = sex_counts_labels.get('f', 0)
m_f_ratio_labels = m_count_labels / f_count_labels if f_count_labels != 0 else float('inf')

# 构建输出内容
output_text = (
    "Subjects included in study \n"
    f"Total 'm': {m_count_1}\n"
    f"Total 'f': {f_count_1}\n"
    f"m/f ratio: {m_f_ratio_1}\n"
    "\n"
    "Re-identifiable Group (class D)\n"
    f"Total 'm': {m_count_d}\n"
    f"Total 'f': {f_count_d}\n"
    f"m/f ratio: {m_f_ratio_d}\n"
    "\n"
    "Subjects used for training\n"
    f"Total 'm': {m_count_2}\n"
    f"Total 'f': {f_count_2}\n"
    f"m/f ratio: {m_f_ratio_2}\n"
    "\n"
    "Subjects used for val_set\n"
    f"Total 'm': {m_count_labels}\n"
    f"Total 'f': {f_count_labels}\n"
    f"m/f ratio: {m_f_ratio_labels}\n"
)

# 将结果保存到一个 .txt 文件
output_file_name = "sex_result.txt"
output_file_path = os.path.join(desktop_path, output_file_name)

with open(output_file_path, 'w') as output_file:
    output_file.write(output_text)

# 打印输出内容
print(output_text)