import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 确认当前工作目录
print("当前工作目录:", os.getcwd())

# 如果当前工作目录不是 kaggle 文件夹，请修改以下路径
train = pd.read_csv(r"data/train.csv")
test = pd.read_csv(r"data/test.csv")

# 设置列名
target_col = "Fertilizer Name"
feature_cols = [col for col in train.columns if col not in ['id', target_col]]

# 确定类别特征
categorical_features = ['Soil Type', 'Crop Type']

# 编码目标变量
le = LabelEncoder()
y_train = le.fit_transform(train[target_col])

# 创建 CatBoost 模型（启用GPU）
model = CatBoostClassifier(
    iterations=300,                # 迭代次数（可调高）
    learning_rate=0.1,             # 学习率
    loss_function='MultiClass',    # 多分类任务
    task_type='GPU',               # GPU加速
    devices='0',                   # 使用第0号GPU
    verbose=50                     # 每50轮打印一次日志
)

# 训练模型
model.fit(train[feature_cols], y_train, cat_features=categorical_features)

# 预测概率
probs = model.predict_proba(test[feature_cols])

# 获取 top 3 预测索引
top3_indices = probs.argsort(axis=1)[:, -3:][:, ::-1]

# 将索引转为类别名称
top3_labels = []
for i in range(top3_indices.shape[1]):
    top3_labels.append(le.inverse_transform(top3_indices[:, i]))

# 转置为 (N, 3) 结构
top3_labels = list(zip(*top3_labels))

# 生成提交字符串
predictions = [' '.join(row) for row in top3_labels]

# 生成提交文件
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': predictions
})
submission.to_csv("submission.csv", index=False)

print("✅ 生成的 submission.csv 示例:")
print(submission.head())
