"""
Train XGBoost salary model and save to models/xgboost_salary.pkl.
Run from project root: python scripts/save_model.py
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

DATA_PROCESSED = pathlib.Path("data/processed")
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("Loading features.parquet ...")
df_feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
X_all = df_feat.drop(columns=["salary_year_avg"])
y_all = df_feat["salary_year_avg"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

print("Training XGBoost ...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

model_path = MODELS_DIR / "xgboost_salary.pkl"
joblib.dump(xgb_model, model_path)
print(f"Model saved → {model_path}")
print(f"Feature columns ({len(X_all.columns)}): {list(X_all.columns)}")
