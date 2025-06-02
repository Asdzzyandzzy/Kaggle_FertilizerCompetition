import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 设置绘图风格
plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)

def feature_engineering_complete(df, train_df=None):
    """
    完整版特征工程 - 包含所有高阶特征
    
    参数:
    df -- 要处理的数据框
    train_df -- 用于计算统计量的训练集 (默认为None)
    """
    # 使用训练集统计量（如果提供）
    if train_df is None:
        train_df = df.copy()
    
    # 深拷贝数据
    df = df.copy()
    
    # === 🌿 基础分箱特征 ===
    df["Temp_bin"] = pd.cut(df["Temparature"], bins=[0,20,25,30,35,40,100], 
                           labels=[0,1,2,3,4,5]).astype("int8")
    
    df["Humidity_bin"] = pd.cut(df["Humidity"], bins=5, labels=False).astype("int8")
    df["Moisture_bin"] = pd.cut(df["Moisture"], bins=5, labels=False).astype("int8")
    
    # === 🌿 类别组合特征 ===
    df["Soil_Crop"] = df["Soil Type"] + "_" + df["Crop Type"]
    df["Soil_Crop_Temp"] = df["Soil_Crop"] + "_" + df["Temp_bin"].astype(str)
    
    # === 🌿 NPK 比率特征 ===
    df["N_P_ratio"] = df["Nitrogen"] / (df["Phosphorous"] + 1e-6)
    df["N_K_ratio"] = df["Nitrogen"] / (df["Potassium"] + 1e-6)
    df["P_K_ratio"] = df["Phosphorous"] / (df["Potassium"] + 1e-6)
    
    # === 🌿 NPK 组合特征 ===
    df["NP_sum"] = df["Nitrogen"] + df["Phosphorous"]
    df["NK_sum"] = df["Nitrogen"] + df["Potassium"]
    df["PK_sum"] = df["Phosphorous"] + df["Potassium"]
    df["NPK_sum"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["NPK_mean"] = df[["Nitrogen", "Phosphorous", "Potassium"]].mean(axis=1)
    df["NPK_std"] = df[["Nitrogen", "Phosphorous", "Potassium"]].std(axis=1)
    df["NPK_skew"] = df[["Nitrogen", "Phosphorous", "Potassium"]].skew(axis=1)
    
    # === 🌿 环境交互特征 ===
    df["Temp_Humidity"] = df["Temparature"] * df["Humidity"] / 100
    df["Temp_Moisture"] = df["Temparature"] * df["Moisture"] / 100
    df["Humidity_Moisture"] = df["Humidity"] * df["Moisture"] / 100
    df["Env_Index"] = df["Temparature"] + df["Humidity"] - df["Moisture"]
    
    # === 🌿 作物和土壤的类别特征 ===
    df["High_N_crop"] = df["Crop Type"].isin(["Sugarcane", "Tobacco", "Cotton"]).astype(int)
    df["High_P_crop"] = df["Crop Type"].isin(["Paddy", "Barley", "Wheat"]).astype(int)
    df["High_K_crop"] = df["Crop Type"].isin(["Fruits", "Vegetables"]).astype(int)
    
    soil_mapping = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3}
    df["Soil_encoded"] = df["Soil Type"].map(soil_mapping)
    
    # === 🌿 类别统计特征 ===
    for col in ["Soil Type", "Crop Type", "Soil_Crop"]:
        for feat in ["Nitrogen", "Phosphorous", "Potassium"]:
            # 仅当分组有足够数据时才计算
            if len(train_df[col].unique()) > 1:
                mean_values = train_df.groupby(col)[feat].mean()
                std_values = train_df.groupby(col)[feat].std().fillna(0)
                df[f"{col}_{feat}_mean"] = df[col].map(mean_values)
                df[f"{col}_{feat}_std"] = df[col].map(std_values)
    
    # === 🌿 时间序列特征 (如果有时间列) ===
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Month'].apply(lambda m: (m%12+3)//3)  # 1:冬, 2:春, 3:夏, 4:秋
        df['Is_rainy_season'] = df['Month'].isin([6,7,8,9]).astype(int)
    
    # === 🌿 多项式特征 ===
    poly_features = ['Temparature', 'Humidity', 'Moisture', 'NPK_sum']
    for feat in poly_features:
        df[f"{feat}_squared"] = df[feat] ** 2
        df[f"{feat}_log"] = np.log1p(df[feat])
    
    return df

def evaluate_features(df, target_columns, models=['lgb', 'xgb', 'cat'], sample_size=None):
    """
    多模型特征评估 - 支持LightGBM, XGBoost和CatBoost
    
    参数:
    df -- 包含特征和目标的数据框
    target_columns -- 目标变量列名列表
    models -- 要使用的模型列表 (默认包含所有)
    sample_size -- 采样大小 (默认为None，使用全部数据)
    """
    # 采样数据（如果指定）
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # 准备数据
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    
    # 分割训练集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 结果容器
    results = {}
    
    # === LightGBM ===
    if 'lgb' in models:
        import lightgbm as lgb
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train.iloc[:,0])
        
        # 训练模型
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        lgb_model = lgb.train(params, train_data, num_boost_round=100)
        
        # 获取特征重要性
        importance = lgb_model.feature_importance(importance_type='gain')
        lgb_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance,
            'Model': 'LightGBM',
            'Target': target_columns[0]
        })
        
        # 添加到结果
        results['LightGBM'] = lgb_importance
    
    # === XGBoost ===
    if 'xgb' in models:
        import xgboost as xgb
        
        # 训练模型
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train.iloc[:,0])
        
        # 获取特征重要性
        importance = xgb_model.feature_importances_
        xgb_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance,
            'Model': 'XGBoost',
            'Target': target_columns[0]
        })
        
        # 添加到结果
        results['XGBoost'] = xgb_importance
    
    # === CatBoost ===
    if 'cat' in models:
        from catboost import CatBoostRegressor
        
        # 识别类别特征
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 训练模型
        cat_model = CatBoostRegressor(
            iterations=100, 
            verbose=False,
            cat_features=cat_features,
            random_state=42
        )
        cat_model.fit(X_train, y_train.iloc[:,0])
        
        # 获取特征重要性
        importance = cat_model.get_feature_importance()
        cat_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance,
            'Model': 'CatBoost',
            'Target': target_columns[0]
        })
        
        # 添加到结果
        results['CatBoost'] = cat_importance
    
    # === 相关性分析 ===
    corr_matrix = df.corr()
    target_corrs = corr_matrix[target_columns]
    
    # === 多重共线性分析 ===
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # 计算VIF需要数值型数据
    num_cols = X.select_dtypes(include=np.number).columns
    vif_data["VIF"] = [variance_inflation_factor(X[num_cols].values, i) 
                      for i in range(len(num_cols))]
    
    # === 可视化结果 ===
    plt.figure(figsize=(16, 20))
    
    # 特征重要性图
    if results:
        all_importance = pd.concat(results.values())
        plt.subplot(3, 1, 1)
        sns.barplot(
            data=all_importance.sort_values('Importance', ascending=False).head(20),
            x='Importance', 
            y='Feature',
            hue='Model'
        )
        plt.title('Top 20 特征重要性 (多模型比较)')
        plt.tight_layout()
    
    # 目标相关性图
    plt.subplot(3, 1, 2)
    sns.heatmap(target_corrs, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('特征与目标变量相关性')
    
    # VIF图
    plt.subplot(3, 1, 3)
    high_vif = vif_data[vif_data['VIF'] > 5].sort_values('VIF', ascending=False)
    sns.barplot(data=high_vif, x='VIF', y='Feature', palette='Reds_r')
    plt.axvline(x=5, color='r', linestyle='--')
    plt.axvline(x=10, color='r', linestyle='--')
    plt.title('高VIF特征 (VIF > 5)')
    
    plt.tight_layout()
    plt.show()
    
    # === 返回结果 ===
    return {
        "feature_importance": all_importance if results else None,
        "target_correlation": target_corrs,
        "vif_scores": vif_data
    }

# ===== 使用示例 =====
# 加载数据
# df = pd.read_csv('your_data.csv')

# 定义目标列
# target_cols = ['Nitrogen', 'Phosphorous', 'Potassium']

# 完整特征工程
# df_full = feature_engineering_complete(df)

# 评估特征 (对每个目标分别评估)
# for target in target_cols:
#     print(f"\n{'='*40}")
#     print(f"评估目标: {target}")
#     print(f"{'='*40}")
#     results = evaluate_features(df_full, [target], models=['lgb', 'xgb', 'cat'], sample_size=10000)

# 查看VIF结果
# vif_df = results['vif_scores']
# print(vif_df.sort_values('VIF', ascending=False).head(10))




# 加载数据
df = pd.read_csv("data/train.csv")  # 替换为你的数据路径

# 特征工程处理
df_full = feature_engineering_complete(df)

# 定义目标列为 Fertilizer Name
target_col = ['Fertilizer Name']

# 对 Fertilizer Name 进行标签编码
le = LabelEncoder()
df_full['Fertilizer_Name_encoded'] = le.fit_transform(df_full['Fertilizer Name'])

# 调用特征评估函数
print("\n==== 评估 Fertilizer Name 的特征影响 ====")
results = evaluate_features(
    df_full.drop(columns=["Fertilizer Name"]),  # 去掉原始标签列
    target_columns=['Fertilizer_Name_encoded'],  # 使用编码后的标签
    models=['lgb', 'xgb', 'cat'],
    sample_size=10000  # 可以调整为你的样本量或 None
)

# 查看 VIF 结果
print("\n==== 高 VIF 特征（潜在多重共线性问题） ====")
vif_df = results['vif_scores']
print(vif_df.sort_values('VIF', ascending=False).head(10))
