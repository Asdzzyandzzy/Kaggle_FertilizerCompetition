import os, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import top_k_accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 1️⃣ 读取数据 ───────────────────────────────────────────
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

TARGET = "Fertilizer Name"
IDCOL  = "id"

# 2️⃣ 高级特征工程 ───────────────────────────────────────
def advanced_fe(df, is_train=True):
    df = df.copy()

    # 比值 / 差分 / 汇总
    df["NP_ratio"]        = df["Nitrogen"] / (df["Phosphorous"] + 1e-3)
    df["NK_ratio"]        = df["Nitrogen"] / (df["Potassium"]   + 1e-3)
    df["PK_ratio"]        = df["Phosphorous"] / (df["Potassium"]+ 1e-3)
    df["Nutrient_sum"]    = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["N_minus_P"]       = df["Nitrogen"] - df["Phosphorous"]
    df["P_minus_K"]       = df["Phosphorous"] - df["Potassium"]

    # 环境交互
    df["Temp_Hum"]        = df["Temparature"] * df["Humidity"] / 100
    df["Moist_Hum"]       = df["Moisture"]    * df["Humidity"] / 100
    df["Env_index"]       = df["Temparature"] + df["Humidity"] - df["Moisture"]

    # 分箱特征
    df["Temp_bin"]        = pd.cut(df["Temparature"], bins=[0,20,25,30,35,40,1e3],
                                   labels=[0,1,2,3,4,5]).astype("int8")
    df["Hum_bin"]         = pd.qcut(df["Humidity"],  q=4, labels=False).astype("int8")
    df["Moist_bin"]       = pd.qcut(df["Moisture"],  q=4, labels=False).astype("int8")

    # 土壤×作物组合
    df["Soil_Crop"]       = df["Soil Type"] + "_" + df["Crop Type"]

    # 简单聚类标签
    df["Nutr_cluster"]    = (df["Nitrogen"]>30).astype(int)*4 + \
                             (df["Phosphorous"]>20).astype(int)*2 + \
                             (df["Potassium"]>10).astype(int)

    # 统计特征（仅用训练集统计，再 merge 回来）
    if is_train:
        stat = df.groupby("Soil Type")["Nitrogen"].agg(["mean","std"]).reset_index()
        stat.columns = ["Soil Type","Soil_N_mean","Soil_N_std"]
        df = df.merge(stat, on="Soil Type", how="left")
    else:
        df = df.merge(soil_stat, on="Soil Type", how="left")

    return df

# 训练集先做，用来求 soil_stat
train_fe = advanced_fe(train, is_train=True)
soil_stat = train_fe[["Soil Type","Soil_N_mean","Soil_N_std"]].drop_duplicates()
test_fe  = advanced_fe(test,  is_train=False)

# 3️⃣ 标签编码 & 特征选择 ────────────────────────────────
cat_cols = ["Soil Type","Crop Type","Soil_Crop","Nutr_cluster"]
num_cols = [c for c in train_fe.columns
            if c not in cat_cols + [TARGET, IDCOL]]

# 目标编码
lbl_y = LabelEncoder()
y = lbl_y.fit_transform(train[TARGET])
N_CLASSES = len(lbl_y.classes_)

# 类别列 LabelEncoding（保持 train / test 同步）
encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    train_fe[col] = enc.fit_transform(train_fe[col])
    test_fe[col]  = enc.transform(test_fe[col])
    encoders[col] = enc

# 特征选择（仅数值列；类别列保留）
selector = SelectKBest(mutual_info_classif, k=min(20, len(num_cols)))
selector.fit(train_fe[num_cols], y)
sel_num_cols = list(np.array(num_cols)[selector.get_support()])

features = sel_num_cols + cat_cols

# 数值特征标准化
scaler = StandardScaler()
train_fe[sel_num_cols] = scaler.fit_transform(train_fe[sel_num_cols])
test_fe[sel_num_cols]  = scaler.transform(test_fe[sel_num_cols])

# 4️⃣ 模型参数 ─────────────────────────────────────────
cat_params = {
    "iterations": 2000, "learning_rate":0.05, "depth":10,
    "l2_leaf_reg":5, "border_count":128,
    "loss_function":"MultiClass", "task_type":"GPU",
    "early_stopping_rounds":100, "verbose":False
}
lgb_params = {
    "objective":"multiclass", "num_class":N_CLASSES, "metric":"multi_logloss",
    "learning_rate":0.05, "num_leaves":128, "max_depth":10,
    "feature_fraction":0.8, "bagging_fraction":0.8, "bagging_freq":5,
    "lambda_l1":0.1, "lambda_l2":0.2, "min_child_samples":20,
    "device":"gpu"
}
xgb_params = {
    "max_depth":8, "learning_rate":0.05, "subsample":0.8,
    "colsample_bytree":0.8, "reg_alpha":0.1, "reg_lambda":0.2,
    "gamma":0.1, "tree_method":"gpu_hist",
    "use_label_encoder":False, "eval_metric":"mlogloss"
}

# 5️⃣ 交叉验证 & 训练 ──────────────────────────────────
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof   = np.zeros((len(train_fe), N_CLASSES))
preds = np.zeros((len(test_fe),  N_CLASSES))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fe, y)):
    print(f"\n▶ Fold {fold+1}/{n_folds}")
    X_tr, X_val = train_fe.iloc[tr_idx][features], train_fe.iloc[val_idx][features]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # CatBoost
    cb = CatBoostClassifier(**cat_params)
    cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    p_cb_val  = cb.predict_proba(X_val)
    p_cb_test = cb.predict_proba(test_fe[features])

    # LightGBM
    lgbm = lgb.LGBMClassifier(**lgb_params, n_estimators=2000)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(200, verbose=False)])
    p_lgb_val  = lgbm.predict_proba(X_val)
    p_lgb_test = lgbm.predict_proba(test_fe[features])

    # XGBoost
    xgbm = xgb.XGBClassifier(**xgb_params, n_estimators=2000, early_stopping_rounds=200)
    xgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb_val  = xgbm.predict_proba(X_val)
    p_xgb_test = xgbm.predict_proba(test_fe[features])

    # 加权集成
    w_val  = 0.5*p_cb_val  + 0.3*p_lgb_val  + 0.2*p_xgb_val
    w_test = 0.5*p_cb_test + 0.3*p_lgb_test + 0.2*p_xgb_test

    oof[val_idx] += w_val
    preds        += w_test / n_folds

    fold_score = top_k_accuracy_score(y_val, w_val, k=3)
    print(f"MAP@3 = {fold_score:.5f}")

# 6️⃣ CV 评分 & 生成提交文件 ───────────────────────────
cv_map3 = top_k_accuracy_score(y, oof, k=3)
print(f"\n★ Overall CV MAP@3 = {cv_map3:.5f}")

# Top-3 结果
top3 = preds.argsort(axis=1)[:, -3:][:, ::-1]
top1, top2, top3 = (lbl_y.inverse_transform(top3[:, i]) for i in range(3))
submission = pd.DataFrame({
    "id": test_fe[IDCOL],
    "Fertilizer Name": [f"{a} {b} {c}" for a, b, c in zip(top1, top2, top3)]
})

# 业务规则后处理
def apply_business_rules(row):
    # if row["Crop Type"] == "Sugarcane" and row["Temparature"] > 35:
    #     return "28-28 14-35-14 10-26-26"
    # if row["Crop Type"] == "Paddy" and row["Phosphorous"] < 15:
    #     return "DAP 14-35-14 10-26-26"
    return None

adjust_mask = test.apply(apply_business_rules, axis=1)
submission.loc[adjust_mask.notna(), "Fertilizer Name"] = adjust_mask[adjust_mask.notna()]

submission.to_csv("submission_final.csv", index=False)
print("✅ submission_final.csv 已生成")
print(submission.head())
