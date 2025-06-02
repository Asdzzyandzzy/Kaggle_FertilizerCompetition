import os, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import top_k_accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ============ 1️⃣ 读取数据 ============ #
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

TARGET = "Fertilizer Name"
IDCOL  = "id"

# ============ 2️⃣ 特征工程 ============ #
def basic_fe(df):
    df = df.copy()
       # === 🌿 必要派生特征（Top20中有依赖的） === #
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

    # === 🌿 分类特征（必要） === #
    df["Soil_Crop"] = df["Soil Type"] + "_" + df["Crop Type"]
    
    return df


train = basic_fe(train)
test  = basic_fe(test)

# ============ 3️⃣ 编码目标变量 ============ #
cat_cols = ["Soil Type", "Crop Type", "Soil_Crop"]
num_cols = [c for c in train.columns if c not in cat_cols + [TARGET, IDCOL]]
le = LabelEncoder()
train["target_enc"] = le.fit_transform(train[TARGET])
N_CLASSES = train["target_enc"].nunique()

# ============ 4️⃣ LightGBM/XGBoost 类别编码 ============ #
lgb_xgb_train = train.copy()
lgb_xgb_test  = test.copy()

for col in cat_cols:
    le_temp = LabelEncoder()
    lgb_xgb_train[col] = le_temp.fit_transform(train[col])
    lgb_xgb_test[col]  = le_temp.transform(test[col])

# ============ 5️⃣ 模型函数 ============ #
def fit_cat(X_tr, y_tr, X_val, y_val, cat_cols, params):
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              cat_features=cat_cols,
              use_best_model=True)
    return model

def fit_lgb(X_tr, y_tr, X_val, y_val, cat_cols, params):
    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    lgb_val   = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, reference=lgb_train)
    bst = lgb.train(params, lgb_train, num_boost_round=3000,
                    valid_sets=[lgb_train, lgb_val],
                    callbacks=[lgb.early_stopping(200)])
    return bst

def fit_xgb(X_tr, y_tr, X_val, y_val, cat_cols, params):
    bst = xgb.XGBClassifier(**params, n_estimators=3000, early_stopping_rounds=200, random_state=42)
    bst.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return bst

# ============ 6️⃣ 最优参数配置 ============ #
best_cat_params = {
    "depth": 6,
    "learning_rate": 0.09463206324241637,
    "l2_leaf_reg": 6.481731902100527,
    "random_strength": 1.2267885514143124,
    "border_count": 193,
    "iterations": 1500,
    "loss_function": "MultiClass",
    "task_type": "GPU",
    "verbose": False,
    "train_dir": None
}


lgb_params = {
    "objective": "multiclass",
    "num_class": 7,
    "metric": "multi_logloss",
    "learning_rate": 0.020256271896515996,
    "num_leaves": 133,
    "max_depth": 8,
    "feature_fraction": 0.6956550642493597,
    "bagging_fraction": 0.6559297289270961,
    "lambda_l1": 0.11129366332111325,
    "lambda_l2": 0.0010269492175026265,
    "device_type": "gpu"
}


xgb_params = {
    "max_depth": 7,
    "learning_rate": 0.0340045505740806,
    "subsample": 0.8883043165516575,
    "colsample_bytree": 0.8091534410913878,
    "reg_alpha": 0.9448099480524449,
    "reg_lambda": 0.1569548785336326,
    "gamma": 1.2576332192970434,
    "tree_method": "gpu_hist",
    "eval_metric": "mlogloss",
    "num_class": 7
}


# ============ 7️⃣ K折交叉验证 ============ #
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred_cat = np.zeros((len(train), N_CLASSES))
oof_pred_lgb = np.zeros_like(oof_pred_cat)
oof_pred_xgb = np.zeros_like(oof_pred_cat)
test_pred_cat = np.zeros((len(test), N_CLASSES))
test_pred_lgb = np.zeros_like(test_pred_cat)
test_pred_xgb = np.zeros_like(test_pred_cat)

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print(f"\n===== Fold {fold+1} / {kf.n_splits} =====")
    X_tr_cat = train.iloc[tr_idx][cat_cols+num_cols];   y_tr = train.iloc[tr_idx]["target_enc"]
    X_val_cat = train.iloc[val_idx][cat_cols+num_cols]; y_val = train.iloc[val_idx]["target_enc"]

    X_tr_lgb = lgb_xgb_train.iloc[tr_idx][cat_cols+num_cols]
    X_val_lgb = lgb_xgb_train.iloc[val_idx][cat_cols+num_cols]

    cat_model = fit_cat(X_tr_cat, y_tr, X_val_cat, y_val, cat_cols, best_cat_params)
    oof_pred_cat[val_idx] = cat_model.predict_proba(X_val_cat)
    test_pred_cat += cat_model.predict_proba(test[cat_cols+num_cols]) / kf.n_splits
    del cat_model; gc.collect()

    lgb_model = fit_lgb(X_tr_lgb, y_tr, X_val_lgb, y_val, cat_cols, lgb_params)
    oof_pred_lgb[val_idx] = lgb_model.predict(X_val_lgb)
    test_pred_lgb += lgb_model.predict(lgb_xgb_test[cat_cols+num_cols]) / kf.n_splits
    del lgb_model; gc.collect()

    xgb_model = fit_xgb(X_tr_lgb, y_tr, X_val_lgb, y_val, cat_cols, xgb_params)
    oof_pred_xgb[val_idx] = xgb_model.predict_proba(X_val_lgb)
    test_pred_xgb += xgb_model.predict_proba(lgb_xgb_test[cat_cols+num_cols]) / kf.n_splits
    del xgb_model; gc.collect()

    fold_map3 = top_k_accuracy_score(y_val, oof_pred_cat[val_idx], k=3)
    print(f"Fold {fold+1} CatBoost MAP@3 = {fold_map3:.4f}")

# ============ 8️⃣ 集成 & 提交 ============ #
oof_pred_ensemble  = 0.5*oof_pred_cat  + 0.3*oof_pred_lgb  + 0.2*oof_pred_xgb
test_pred_ensemble = 0.5*test_pred_cat + 0.3*test_pred_lgb + 0.2*test_pred_xgb

cv_map3 = top_k_accuracy_score(train["target_enc"], oof_pred_ensemble, k=3)
print(f"\n===== 5-Fold CV MAP@3 (Ensemble) = {cv_map3:.5f} =====\n")

top3_idx = test_pred_ensemble.argsort(axis=1)[:, -3:][:, ::-1]
col1 = le.inverse_transform(top3_idx[:, 0])
col2 = le.inverse_transform(top3_idx[:, 1])
col3 = le.inverse_transform(top3_idx[:, 2])
pred_strings = pd.Series(col1) + " " + pd.Series(col2) + " " + pd.Series(col3)

submission = pd.DataFrame({
    "id": test[IDCOL],
    "Fertilizer Name": pred_strings
})

submission.to_csv("submission4.csv", index=False)
print("✅ Submission saved to submission.csv")
print(submission.head())
