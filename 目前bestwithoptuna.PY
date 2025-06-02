import gc, warnings, numpy as np, pandas as pd, optuna, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import top_k_accuracy_score

warnings.filterwarnings("ignore")

# ========== 0. 全局配置 ==========
TARGET, IDCOL = "Fertilizer Name", "id"
N_FOLDS, N_TRIALS, SEED = 5, 15, 2005
CAT_ITERS_BASE, LGB_ITERS_BASE, XGB_ITERS_BASE = 1500, 2500, 2500
CV_ESR = 200

np.random.seed(SEED)

# ========== 1. 数据读取 ==========
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

# ========== 2. 特征工程 ==========
# ========== 2. 特征工程 ========== #
def build_features(df):
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


train = build_features(train)
test  = build_features(test)

cat_cols = ["Soil Type","Crop Type","Soil_Crop"]
num_cols = [c for c in train.columns if c not in cat_cols + [TARGET, IDCOL]]
le_y = LabelEncoder()
y = le_y.fit_transform(train[TARGET])
N_CLASSES = len(le_y.classes_)

for col in cat_cols:
    enc = LabelEncoder()
    train[col] = enc.fit_transform(train[col])
    test[col]  = enc.transform(test[col])

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols]  = scaler.transform(test[num_cols])
X_all = train[cat_cols+num_cols]

# ========== 3. Optuna 调参 ==========
def cat_objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 6, 10),
        "learning_rate": trial.suggest_float("lr", 0.02, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2", 1, 10, log=True),
        "random_strength": trial.suggest_float("rs", 0.1, 3.0),
        "border_count": trial.suggest_int("border", 64, 255),
        "iterations": CAT_ITERS_BASE,
        "loss_function": "MultiClass",
        "task_type": "GPU",
        "verbose": False
    }
    cv = CatBoostClassifier(**params, train_dir=f"catboost_info/trial_{trial.number}")
    skf = StratifiedKFold(3, shuffle=True, random_state=trial.number)
    scores=[]
    for tr,va in skf.split(X_all,y):
        cv.fit(X_all.iloc[tr], y[tr], eval_set=(X_all.iloc[va], y[va]), cat_features=cat_cols, early_stopping_rounds=100, verbose=False)
        scores.append(top_k_accuracy_score(y[va], cv.predict_proba(X_all.iloc[va]), k=3))
    return 1 - np.mean(scores)

def lgb_objective(trial):
    params = {
        "objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("lr",0.02,0.15,log=True),
        "num_leaves": trial.suggest_int("leaves",64,256),
        "max_depth": trial.suggest_int("depth",-1,12),
        "feature_fraction": trial.suggest_float("ff",0.6,0.95),
        "bagging_fraction": trial.suggest_float("bf",0.6,0.95),
        "lambda_l1": trial.suggest_float("l1",1e-3,1,log=True),
        "lambda_l2": trial.suggest_float("l2",1e-3,1,log=True),
        "device_type": "gpu"
    }
    skf = StratifiedKFold(3, shuffle=True, random_state=trial.number)
    scores=[]
    for tr,va in skf.split(X_all,y):
        tr_ds = lgb.Dataset(X_all.iloc[tr], y[tr], categorical_feature=cat_cols)
        va_ds = lgb.Dataset(X_all.iloc[va], y[va], categorical_feature=cat_cols)
        booster = lgb.train(params, tr_ds, num_boost_round=LGB_ITERS_BASE, valid_sets=[va_ds], callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(top_k_accuracy_score(y[va], booster.predict(X_all.iloc[va]), k=3))
    return 1 - np.mean(scores)

def xgb_objective(trial):
    params = {
        "max_depth": trial.suggest_int("depth",6,12),
        "learning_rate": trial.suggest_float("lr",0.02,0.15,log=True),
        "subsample": trial.suggest_float("sub",0.6,1.0),
        "colsample_bytree": trial.suggest_float("col",0.6,1.0),
        "reg_alpha": trial.suggest_float("alpha",1e-3,1,log=True),
        "reg_lambda": trial.suggest_float("lam",1e-3,1,log=True),
        "gamma": trial.suggest_float("gamma",0,2),
        "tree_method": "gpu_hist",
        "eval_metric": "mlogloss",
        "num_class": N_CLASSES,
        "verbosity": 0
    }
    skf = StratifiedKFold(3, shuffle=True, random_state=trial.number)
    scores=[]
    for tr,va in skf.split(X_all,y):
        xgbm = xgb.XGBClassifier(**params, n_estimators=XGB_ITERS_BASE, early_stopping_rounds=100, random_state=trial.number)
        xgbm.fit(X_all.iloc[tr], y[tr], eval_set=[(X_all.iloc[va], y[va])], verbose=False)
        scores.append(top_k_accuracy_score(y[va], xgbm.predict_proba(X_all.iloc[va]), k=3))
    return 1 - np.mean(scores)

print("⏳ Optuna tuning CatBoost ...")
study_cb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=False)
cat_best = study_cb.best_params
cat_best["learning_rate"] = cat_best.pop("lr")
cat_best.update({"iterations": CAT_ITERS_BASE, "loss_function": "MultiClass", "task_type": "GPU", "verbose": False})
print("✅ CatBoost best params:", cat_best)

print("⏳ Optuna tuning LightGBM ...")
study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
lgb_best = study_lgb.best_params
lgb_best.update({"objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss", "device_type": "gpu"})
print("✅ LightGBM best params:", lgb_best)

print("⏳ Optuna tuning XGBoost ...")
study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
xgb_best = study_xgb.best_params
xgb_best.update({"tree_method": "gpu_hist", "eval_metric": "mlogloss", "num_class": N_CLASSES})
print("✅ XGBoost best params:", xgb_best)

print("✅ CatBoost best params:", cat_best)
print("✅ LightGBM best params:", lgb_best)
print("✅ XGBoost best params:", xgb_best)
# 🚀 完结
print("✅ All tuning done, ready for 5-Fold training!")

#✅ LightGBM best params: {'lr': 0.020256271896515996, 'leaves': 133, 'depth': 8, 'ff': 0.6956550642493597, 'bf': 0.6559297289270961, 'l1': 0.11129366332111325, 'l2': 0.0010269492175026265, 'objective': 'multiclass', 'num_class': 7, 'metric': 'multi_logloss', 'device_type': 'gpu'}

#✅ CatBoost best params: {'depth': 6, 'l2': 6.481731902100527, 'rs': 1.2267885514143124, 'border': 193, 'learning_rate': 0.09463206324241637, 'iterations': 1500, 'loss_function': 'MultiClass', 'task_type': 'GPU', 'verbose': False}
#✅ LightGBM best params: {'lr': 0.020256271896515996, 'leaves': 133, 'depth': 8, 'ff': 0.6956550642493597, 'bf': 0.6559297289270961, 'l1': 0.11129366332111325, 'l2': 0.0010269492175026265, 'objective': 'multiclass', 'num_class': 7, 'metric': 'multi_logloss', 'device_type': 'gpu'}
#✅ XGBoost best params: {'depth': 7, 'lr': 0.0340045505740806, 'sub': 0.8883043165516575, 'col': 0.8091534410913878, 'alpha': 0.9448099480524449, 'lam': 0.1569548785336326, 'gamma': 1.2576332192970434, 'tree_method': 'gpu_hist', 'eval_metric': 'mlogloss', 'num_class': 7}

