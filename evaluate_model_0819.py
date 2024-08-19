import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# 获取当前用户桌面的路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 指定模型和数据文件的路径
model_path = os.path.join(desktop_path, 'final_random_forest_model.joblib')
heldout_data_file_path = os.path.join(desktop_path, 'EEG_Cleaned_Features_heldout', 'heldout_features.xlsx')
extracted_data_file_path = os.path.join(desktop_path, 'extracted_features.xlsx')

# 加载模型
loaded_model = joblib.load(model_path)

# 加载heldout数据集
heldout_df = pd.read_excel(heldout_data_file_path)

# 分离heldout数据集中的特征和标签
X_heldout = heldout_df.iloc[:, :-1]
y_heldout = heldout_df.iloc[:, -1]

# 使用加载的模型进行预测
y_heldout_pred = loaded_model.predict(X_heldout)

# 计算正确率
accuracy = accuracy_score(y_heldout, y_heldout_pred)
print(f"模型在 heldout 数据集上的预测正确率: {accuracy:.4f}")

# 加载extracted_features数据集
extracted_df = pd.read_excel(extracted_data_file_path)

# 提取extracted_features数据集中的标签
extracted_labels = extracted_df.iloc[:, -1]

# 查找heldout_features数据集中哪些标签也在extracted_features的标签中
common_labels_mask = y_heldout.isin(extracted_labels)
common_labels = y_heldout[common_labels_mask]
X_common = X_heldout[common_labels_mask]

# 基于这些label的 heldout_features.xlsx 中的特征再次作为输入，用模型进行预测
y_common_pred = loaded_model.predict(X_common)

# 计算准确度
common_accuracy = accuracy_score(common_labels, y_common_pred)

# 输出真实标签、预测标签以及准确度
output_df = pd.DataFrame({
    '真实标签': common_labels,
    '预测标签': y_common_pred
})
print("\n基于共同标签的预测结果：")
print(output_df)
print(f"\n模型在共同标签上的预测准确度: {common_accuracy:.4f}")
