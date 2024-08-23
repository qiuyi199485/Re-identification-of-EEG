import pandas as pd

# 读取Excel文件，指定路径
input_file_path = 'C:\\Users\\49152\\Desktop\\val_set_feature_3.pkl'
df = pd.read_pickle(input_file_path)

# 保存为Pickle文件，指定路径
output_file_path = 'C:\\Users\\49152\\Desktop\\val_set_features.xlsx'
df.to_excel(output_file_path)
