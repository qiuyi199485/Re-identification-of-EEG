import matplotlib.pyplot as plt
import seaborn as sns
import os
import settings
import sys
import mne
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from autoreject import AutoReject
from scipy.signal import welch
from scipy.stats import kurtosis, skew
sys.path.insert( 1,'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')                              # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
#sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')       

from settings import f_s, f_min, f_max
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 加载数据
train_df = pd.read_excel('~/Desktop/extracted_features.xlsx')
val_df = pd.read_excel('~/Desktop/val_set.xlsx')
test_df = pd.read_excel('~/Desktop/test_set.xlsx')

# 检查并删除缺失值
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# 分离特征和标签
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_val = val_df.iloc[:, :-1]
y_val = val_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# 定义目标函数
def objective(trial):
    # 定义随机森林的超参数空间
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # 创建随机森林模型
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # 在训练集上训练模型
    model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    return accuracy

# 保存每次迭代的准确率
accuracy_history = []

# 创建并优化 study
def callback(study, trial):
    accuracy_history.append(trial.value)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, callbacks=[callback])

# 输出最优参数和得分
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 使用最优参数训练最终模型
best_params = study.best_params
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train, y_train)

# 在测试集上进行预测
y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

# 注意：roc_auc_score对于多分类问题需要特殊处理，以下是平均策略
roc_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted')

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")
print(f"Test AUC-ROC: {roc_auc}")

# 获取特征重要性并显示前二十个
feature_importances = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
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
plt.title("Random_Forest Feature Selection")
plt.xlabel("Relative importance")
plt.ylabel("Features")
plt.show()


# Save model
model_path = os.path.join(desktop_path, 'final_random_forest_model.joblib')
joblib.dump(final_model, model_path)
print(f"model saved to : {model_path}")


# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o')
plt.title('Learning Curve (Accuracy over Trials)')
plt.xlabel('Trial')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()