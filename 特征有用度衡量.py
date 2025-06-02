import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === 数据加载 ===
df = pd.read_csv("data/train.csv")  # 替换为你的路径
print("✅ 数据加载完成，样本数：", len(df))

# === 特征工程函数 ===
def feature_engineering_complete(df):
    df = df.copy()
    df["Temp_bin"] = pd.cut(df["Temparature"], bins=[0,20,25,30,35,40,100], labels=[0,1,2,3,4,5]).astype("int8")
    df["Humidity_bin"] = pd.cut(df["Humidity"], bins=5, labels=False).astype("int8")
    df["Moisture_bin"] = pd.cut(df["Moisture"], bins=5, labels=False).astype("int8")
    df["Soil_Crop"] = df["Soil Type"] + "_" + df["Crop Type"]
    df["N_P_ratio"] = df["Nitrogen"] / (df["Phosphorous"] + 1e-6)
    df["N_K_ratio"] = df["Nitrogen"] / (df["Potassium"] + 1e-6)
    df["P_K_ratio"] = df["Phosphorous"] / (df["Potassium"] + 1e-6)
    df["NPK_sum"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["NPK_mean"] = df[["Nitrogen", "Phosphorous", "Potassium"]].mean(axis=1)
    df["NPK_std"] = df[["Nitrogen", "Phosphorous", "Potassium"]].std(axis=1)
    df["Temp_Humidity"] = df["Temparature"] * df["Humidity"] / 100
    df["Temp_Moisture"] = df["Temparature"] * df["Moisture"] / 100
    df["Humidity_Moisture"] = df["Humidity"] * df["Moisture"] / 100
    df["Env_Index"] = df["Temparature"] + df["Humidity"] - df["Moisture"]
    df["High_N_crop"] = df["Crop Type"].isin(["Sugarcane", "Tobacco", "Cotton"]).astype(int)
    df["High_P_crop"] = df["Crop Type"].isin(["Paddy", "Barley", "Wheat"]).astype(int)
    for col in ["Soil Type", "Crop Type", "Soil_Crop"]:
        for feat in ["Nitrogen", "Phosphorous", "Potassium"]:
            if col in df.columns:
                mean_values = df.groupby(col)[feat].mean()
                std_values = df.groupby(col)[feat].std().fillna(0)
                df[f"{col}_{feat}_mean"] = df[col].map(mean_values)
                df[f"{col}_{feat}_std"] = df[col].map(std_values)
    return df

# 特征工程处理
df_full = feature_engineering_complete(df)
print("✅ 特征工程完成，特征数量：", df_full.shape[1])

# Fertilizer Name 编码
le = LabelEncoder()
df_full['Fertilizer_Name_encoded'] = le.fit_transform(df_full['Fertilizer Name'])

# === 特征评估 ===
X = df_full.drop(columns=["Fertilizer Name", "Fertilizer_Name_encoded", "id"])

# 🚀 自动处理所有类别列
for col in X.select_dtypes(include=['object', 'category']).columns:
    le_temp = LabelEncoder()
    X[col] = le_temp.fit_transform(X[col].astype(str))

y = df_full["Fertilizer_Name_encoded"]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X.fillna(0), y)

importances = rf.feature_importances_
feat_importance = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_importance = feat_importance.sort_values(by="Importance", ascending=False)

print("\n=== Top 20 特征重要性 ===")
print(feat_importance.head(20))

# 可视化
plt.figure(figsize=(10,8))
sns.barplot(data=feat_importance.head(20), x="Importance", y="Feature")
plt.title("Top 20 特征对 Fertilizer Name 的重要性")
plt.tight_layout()
plt.show()

# === VIF 多重共线性分析 ===
num_cols = X.select_dtypes(include=np.number).columns
vif_data = pd.DataFrame()
vif_data["Feature"] = num_cols
vif_data["VIF"] = [variance_inflation_factor(X[num_cols].values, i) for i in range(len(num_cols))]

print("\n=== 高 VIF 特征 (多重共线性) ===")
print(vif_data.sort_values("VIF", ascending=False).head(10))
