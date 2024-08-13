import pandas as pd

# 读取Excel文件，指定路径
input_file_path = 'C:\\Users\\49152\\Desktop\\MA\\MA\\dataframe_all.xlsx'
df = pd.read_excel(input_file_path)

# 保存为Pickle文件，指定路径
output_file_path = 'C:\\Users\\49152\\Desktop\\dataframe_all.pkl'
df.to_pickle(output_file_path)
