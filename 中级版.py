# run_flagship.py  ─────────────────────────────────────────────────────────
import gc, os, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import top_k_accuracy_score
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ========== 0. 全局参数 ==========
SEEDS          = [42, 2025]          # 多随机种子 Bagging
N_FOLDS        = 5
CAT_WEIGHT, LGB_WEIGHT, XGB_WEIGHT = 0.5, 0.3, 0.2
CAT_ITERS, LGB_ITERS, XGB_ITERS     = 1500, 2500, 2500

# ========== 1. 读取数据 ==========
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
TARGET, IDCOL = "Fertilizer Name", "id"

# ========== 2. 高级特征工程 ==========
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 基础分箱
    df["Temp_bin"]     = pd.cut(df["Temparature"],
                                bins=[0,20,25,30,35,40,1e6],
                                labels=[0,1,2,3,4,5]).astype("int8")
    df["Hum_bin"]      = pd.qcut(df["Humidity"], 4, labels=False).astype("int8")
    df["Moist_bin"]    = pd.qcut(df["Moisture"], 4, labels=False).astype("int8")

    # 营养元素比值 & 差分
    df["NP_ratio"]     = df["Nitrogen"] / (df["Phosphorous"] + 1e-3)
    df["NK_ratio"]     = df["Nitrogen"] / (df["Potassium"]   + 1e-3)
    df["PK_ratio"]     = df["Phosphorous"] / (df["Potassium"]+ 1e-3)
    df["Nutrient_sum"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["N_minus_P"]    = df["Nitrogen"] - df["Phosphorous"]
    df["P_minus_K"]    = df["Phosphorous"] - df["Potassium"]

    # 环境交互
    df["Temp_Hum"]     = df["Temparature"] * df["Humidity"]  / 100
    df["Moist_Hum"]    = df["Moisture"]    * df["Humidity"]  / 100
    df["Env_index"]    = df["Temparature"] + df["Humidity"] - df["Moisture"]

    # 交叉类别
    df["Soil_Crop"]    = df["Soil Type"] + "_" + df["Crop Type"]

    return df

train = build_features(train)
test  = build_features(test)

# ========== 3. 编码与列清单 ==========
cat_cols = ["Soil Type","Crop Type","Soil_Crop"]
num_cols = [c for c in train.columns if c not in cat_cols + [TARGET, IDCOL]]

# 目标标签
lbl_y = LabelEncoder()
y = lbl_y.fit_transform(train[TARGET])
N_CLASSES = len(lbl_y.classes_)

# 类别列编码一致化
encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    train[col] = enc.fit_transform(train[col])
    test[col]  = enc.transform(test[col])
    encoders[col] = enc

# 数值列标准化
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols]  = scaler.transform(test[num_cols])

features = num_cols + cat_cols

# ========== 4. 模型参数 ==========
cat_params = dict(iterations=CAT_ITERS, learning_rate=0.05, depth=8,
                  l2_leaf_reg=3, border_count=128,
                  loss_function="MultiClass", task_type="GPU",
                  early_stopping_rounds=200, verbose=False)

lgb_params = dict(objective="multiclass", num_class=N_CLASSES,
                  metric="multi_logloss", learning_rate=0.05,
                  num_leaves=128, max_depth=-1,
                  feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=5,
                  lambda_l1=0.1, lambda_l2=0.2, device_type="gpu")

xgb_params = dict(max_depth=8, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2, gamma=0.1,
                  tree_method="gpu_hist", use_label_encoder=False,
                  eval_metric="mlogloss")

# ========== 5. 训练 & Bagging ==========
oof_all = np.zeros((len(train), N_CLASSES))
pred_all= np.zeros((len(test) , N_CLASSES))

for seed in SEEDS:
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof_seed = np.zeros_like(oof_all)
    pred_seed= np.zeros_like(pred_all)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        X_tr, y_tr = train.iloc[tr_idx][features], y[tr_idx]
        X_va, y_va = train.iloc[va_idx][features], y[va_idx]

        # CatBoost
        cat_params["random_seed"] = seed
        cb = CatBoostClassifier(**cat_params)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        p_cb_va  = cb.predict_proba(X_va)
        p_cb_te  = cb.predict_proba(test[features])

        # LightGBM
        lgbm = lgb.LGBMClassifier(**lgb_params, n_estimators=LGB_ITERS, random_state=seed)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                 callbacks=[lgb.early_stopping(200, verbose=False)])
        p_lgb_va = lgbm.predict_proba(X_va)
        p_lgb_te = lgbm.predict_proba(test[features])

        # XGBoost
        xgbm = xgb.XGBClassifier(**xgb_params, n_estimators=XGB_ITERS,
                                 early_stopping_rounds=200, random_state=seed)
        xgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb_va = xgbm.predict_proba(X_va)
        p_xgb_te = xgbm.predict_proba(test[features])

        # 加权融合
        w_va  = (CAT_WEIGHT*p_cb_va  +
                 LGB_WEIGHT*p_lgb_va +
                 XGB_WEIGHT*p_xgb_va)
        w_te  = (CAT_WEIGHT*p_cb_te  +
                 LGB_WEIGHT*p_lgb_te +
                 XGB_WEIGHT*p_xgb_te)

        oof_seed[va_idx] += w_va
        pred_seed        += w_te / N_FOLDS

        print(f"Seed {seed} Fold {fold+1}: MAP@3 =",
              top_k_accuracy_score(y_va, w_va, k=3).round(5))

        del cb, lgbm, xgbm; gc.collect()

    # 元模型 Stacking（Ridge）
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(oof_seed, y)
    oof_stacked  = ridge.decision_function(oof_seed)
    pred_stacked = ridge.decision_function(pred_seed)

    # Softmax
    oof_prob  = np.exp(oof_stacked)  / np.exp(oof_stacked).sum(1, keepdims=True)
    pred_prob = np.exp(pred_stacked) / np.exp(pred_stacked).sum(1, keepdims=True)

    oof_all  += oof_prob  / len(SEEDS)
    pred_all += pred_prob / len(SEEDS)

# ========== 6. 评估 & 输出 ==========
cv_map3 = top_k_accuracy_score(y, oof_all, k=3)
print(f"\n★★★★  Overall CV MAP@3 = {cv_map3:.5f} ★★★★")

top3 = pred_all.argsort(axis=1)[:, -3:][:, ::-1]
t1, t2, t3 = (lbl_y.inverse_transform(top3[:, i]) for i in range(3))
submission = pd.DataFrame({
    "id": test[IDCOL],
    "Fertilizer Name": [f"{a} {b} {c}" for a, b, c in zip(t1, t2, t3)]
})
submission.to_csv("submission_flagship.csv", index=False)
print("✅ submission_flagship.csv saved")
print(submission.head())
