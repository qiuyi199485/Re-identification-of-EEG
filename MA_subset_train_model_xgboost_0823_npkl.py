import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter


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
test_folder = os.path.expanduser('~/Desktop/Feature_test')

train_df = load_pickle_files_from_folder(train_folder)
test_df = load_pickle_files_from_folder(test_folder)

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
    
    # 使用交叉验证评估模型
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, train_X, train_y_encoded, cv=cv, scoring='accuracy').mean()
    
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

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")

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

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o')
plt.title('Learning Curve (Accuracy over Trials)')
plt.xlabel('Trial')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()
