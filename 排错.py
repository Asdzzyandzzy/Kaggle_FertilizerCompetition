import xgboost as xgb
import pandas as pd
import numpy as np

X = pd.DataFrame({
    "Soil Type": [0, 1, 2, 0, 1],
    "Crop Type": [1, 0, 1, 0, 2],
    "Soil_Crop": [0, 1, 2, 1, 0],
    "Temparature": [25, 30, 28, 35, 32],
    "Humidity": [60, 65, 70, 55, 50]
})
y = np.array([0, 1, 2, 1, 0])

model = xgb.XGBClassifier(
    max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    n_estimators=100,
    num_class=3,
    tree_method="gpu_hist",
    random_state=42,
    early_stopping_rounds=10
)
model.fit(X, y, eval_set=[(X, y)], verbose=False)
print("✅ XGBoost 运行成功！")
