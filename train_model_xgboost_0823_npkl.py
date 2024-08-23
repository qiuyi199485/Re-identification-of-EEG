import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter


#xgb.set_config(verbosity=3)
xgb.set_config(verbosity=0)


# 定义一个函数来加载特定文件夹中的所有pickle文件并将其合并成一个DataFrame
def load_pickle_files_from_folder(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
    df_list = []
    for file in all_files:
        with open(file, 'rb') as f:
            df_list.append(pickle.load(f))
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df



# 加载数据
train_folder = os.path.expanduser('~/Desktop/Feature_train')
val_folder = os.path.expanduser('~/Desktop/Feature_val')
test_folder = os.path.expanduser('~/Desktop/Feature_test')

train_df = load_pickle_files_from_folder(train_folder)
val_df = load_pickle_files_from_folder(val_folder)
test_df = load_pickle_files_from_folder(test_folder)

# get fratures and labels
train_X = train_df.iloc[:, :-1]  # 除最后一列之外的所有列（特征）
train_y = train_df.iloc[:, -1]   # 最后一列（字符串类型标签）

val_X = val_df.iloc[:, :-1]      # 验证集特征
val_y = val_df.iloc[:, -1]       # 验证集标签

test_X = test_df.iloc[:, :-1]    # 测试集特征
test_y = test_df.iloc[:, -1]     # 测试集标签

label_encoder = LabelEncoder()

train_y_encoded = label_encoder.fit_transform(train_y)
val_y_encoded = label_encoder.transform(val_y)
test_y_encoded = label_encoder.transform(test_y)

object_cols = train_X.select_dtypes(include=['object']).columns
#print("Object columns in the DataFrame:", object_cols)

# 尝试将这些列转换为数值类型
for col in object_cols:
    train_X[col] = pd.to_numeric(train_X[col], errors='coerce')
    val_X[col] = pd.to_numeric(val_X[col], errors='coerce')
    test_X[col] = pd.to_numeric(test_X[col], errors='coerce')
    
# 创建 DMatrix
#dtrain = xgb.DMatrix(data=train_X, label=train_y_encoded)
#dval = xgb.DMatrix(data=val_X, label=val_y_encoded)
#dtest = xgb.DMatrix(data=test_X, label=test_y_encoded)


# 定义目标函数
def objective(trial):
    # 定义 XGBoost 的超参数空间
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 32)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    # 创建 XGBoost 模型
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist',  # 使用GPU加速的tree方法
        device='cuda'  # 使用第一个GPU
    )
    
    # 在训练集上训练模型
    model.fit(train_X, train_y_encoded)
    
    # 在验证集上评估模型
    y_val_pred = model.predict(val_X)
    accuracy = accuracy_score(val_y_encoded, y_val_pred)
    
    return accuracy

# 保存每次迭代的准确率
accuracy_history = []

# 创建并优化 study
def callback(study, trial):
    accuracy_history.append(trial.value)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[callback])

# 输出最优参数和得分
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 使用最优参数训练最终模型
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='mlogloss')
final_model.fit(train_X, train_y_encoded)

# 在测试集上进行预测
y_test_pred = final_model.predict(test_X)
y_test_prob = final_model.predict_proba(test_X)

# 评估模型性能
accuracy = accuracy_score(test_y_encoded, y_test_pred)
precision = precision_score(test_y_encoded, y_test_pred, average='weighted')
recall = recall_score(test_y_encoded, y_test_pred, average='weighted')
f1 = f1_score(test_y_encoded, y_test_pred, average='weighted')

# 注意：roc_auc_score对于多分类问题需要特殊处理，以下是平均策略
#roc_auc = roc_auc_score(test_y_encoded, y_test_prob, multi_class='ovr', average='weighted')

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")
#print(f"Test AUC-ROC: {roc_auc}")

# 获取特征重要性并显示前二十个
feature_importances = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': train_X.columns,
    'Importance': feature_importances
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

# Save model
#desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
#model_path = os.path.join(desktop_path, 'final_xgboost_model.joblib')
#joblib.dump(final_model, model_path)
#print(f"Model saved to : {model_path}")

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o')
plt.title('Learning Curve (Accuracy over Trials)')
plt.xlabel('Trial')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()

# 对于每个label计算投票后的预测结果
def voting_prediction_per_label(model, X_test, y_test):
    unique_labels = np.unique(y_test)
    final_predictions = []
    for label in unique_labels:
        # 找到所有真实标签为label的样本
        label_indices = np.where(y_test == label)[0]
        label_samples = X_test.iloc[label_indices]

        # 使用模型预测这些样本的标签
        label_predictions = model.predict(label_samples)

        # 进行投票，选择出现次数最多的预测标签作为最终的预测结果
        most_common_label = Counter(label_predictions).most_common(1)[0][0]

        # 保存最终的预测结果
        final_predictions.append(most_common_label)
    
    return final_predictions, unique_labels

# 使用投票机制在测试集上进行预测
final_predictions_volt, unique_labels_volt = voting_prediction_per_label(final_model, test_X, test_y_encoded)

# 将投票后的预测结果与真实的标签进行比较
accuracy_volt = accuracy_score(unique_labels_volt, final_predictions_volt)
precision_volt = precision_score(unique_labels_volt, final_predictions_volt, average='weighted')
recall_volt = recall_score(unique_labels_volt, final_predictions_volt, average='weighted')
f1_volt = f1_score(unique_labels_volt, final_predictions_volt, average='weighted')

print("Voting-based Test Accuracy: ", accuracy_volt)
print("Voting-based Test Precision: ", precision_volt)
print("Voting-based Test Recall: ", recall_volt)
print("Voting-based Test F1 Score: ", f1_volt)