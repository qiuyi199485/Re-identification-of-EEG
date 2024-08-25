import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter

xgb.set_config(verbosity=0)

# 定义文件路径
train_file_path = os.path.expanduser('~/Desktop/session_first_feature.pkl')
test_file_path = os.path.expanduser('~/Desktop/session_second_feature.pkl')

# 从pickle文件中加载数据
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 加载数据
train_df = load_pickle_file(train_file_path)
test_df = load_pickle_file(test_file_path)
# 获取特征和标签
train_X = train_df.iloc[:, :-1]  # 除最后一列之外的所有列（特征）
train_y = train_df.iloc[:, -1]   # 最后一列（字符串类型标签）

test_X = test_df.iloc[:, :-1]    # 测试集特征
test_y = test_df.iloc[:, -1]     # 测试集标签

label_encoder = LabelEncoder()

train_y_encoded = label_encoder.fit_transform(train_y)
test_y_encoded = label_encoder.transform(test_y)

object_cols = train_X.select_dtypes(include=['object']).columns

# 尝试将这些列转换为数值类型
for col in object_cols:
    train_X[col] = pd.to_numeric(train_X[col], errors='coerce')
    test_X[col] = pd.to_numeric(test_X[col], errors='coerce')

# 划分训练集为训练子集和验证子集
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y_encoded, test_size=0.2, random_state=42, stratify=train_y_encoded)

# 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_X, label=test_y_encoded)

# 固定的超参数
fixed_params = {
    'max_depth': 3,
    'learning_rate': 0.03661037033338074,
    #'learning_rate': 0.03,
    'subsample': 0.67386985144832,
    'colsample_bytree': 0.8818101726706227,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',  # 使用GPU加速的tree方法
    'num_class': len(np.unique(train_y_encoded)),
    'seed': 42
}

# 训练模型
evals_result = {}
bst = xgb.train(
    fixed_params, 
    dtrain, 
    num_boost_round=938,  # 设置一个较大的值，用于手动观察收敛曲线
    evals=[(dval, 'validation')],
    evals_result=evals_result,
    verbose_eval=True
)

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(evals_result['validation']['mlogloss'], label='Validation Log Loss')
plt.title('Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()

# 在测试集上进行预测
y_test_prob = bst.predict(dtest)
y_test_pred = np.argmax(y_test_prob, axis=1)

# 评估模型性能
accuracy = accuracy_score(test_y_encoded, y_test_pred)
precision = precision_score(test_y_encoded, y_test_pred, average='weighted')
recall = recall_score(test_y_encoded, y_test_pred, average='weighted')
f1 = f1_score(test_y_encoded, y_test_pred, average='weighted')

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")

# 计算和显示归一化的特征重要性
feature_importances = bst.get_score(importance_type='weight')
total_importance = sum(feature_importances.values())
importance_df = pd.DataFrame({
    'Feature': list(feature_importances.keys()),
    'Importance': [v / total_importance for v in feature_importances.values()]
}).sort_values(by='Importance', ascending=False)

# 打印前20个重要特征
print("Top 20 Important Features:")
print(importance_df.head(20))

# 可视化前20个重要特征
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df.head(20),
    palette=sns.color_palette("coolwarm_r", len(importance_df.head(20)))  # 使用渐变色
)
plt.title("XGBoost Feature Importance")
plt.xlabel("Relative importance")
plt.ylabel("Features")
plt.show()

# 对于每个label计算投票后的预测结果
def voting_prediction_per_label(model, X_test, y_test):
    unique_labels = np.unique(y_test)
    final_predictions = []
    for label in unique_labels:
        # 找到所有真实标签为label的样本
        label_indices = np.where(y_test == label)[0]
        label_samples = X_test.iloc[label_indices]

        # 将样本转换为DMatrix格式
        dmatrix_samples = xgb.DMatrix(label_samples)

        # 使用模型预测这些样本的标签
        label_predictions = model.predict(dmatrix_samples)
        label_predictions = np.argmax(label_predictions, axis=1)

        # 进行投票，选择出现次数最多的预测标签作为最终的预测结果
        most_common_label = Counter(label_predictions).most_common(1)[0][0]

        # 保存最终的预测结果
        final_predictions.append(most_common_label)
    
    return final_predictions, unique_labels

# 使用投票机制在测试集上进行预测
final_predictions_volt, unique_labels_volt = voting_prediction_per_label(bst, test_X, test_y_encoded)

# 将投票后的预测结果与真实的标签进行比较
accuracy_volt = accuracy_score(unique_labels_volt, final_predictions_volt)
precision_volt = precision_score(unique_labels_volt, final_predictions_volt, average='weighted')
recall_volt = recall_score(unique_labels_volt, final_predictions_volt, average='weighted')
f1_volt = f1_score(unique_labels_volt, final_predictions_volt, average='weighted')

print("Voting-based Test Accuracy: ", accuracy_volt)
print("Voting-based Test Precision: ", precision_volt)
print("Voting-based Test Recall: ", recall_volt)
print("Voting-based Test F1 Score: ", f1_volt)