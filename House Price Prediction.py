#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ================================================
# TASK 3: HOUSE PRICE PREDICTION (COMPLETE CODE)
# ================================================


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------------------
# Step 1 — Load Dataset 
# -------------------------------------------

csv_path = r"C:\Users\HP\OneDrive\Desktop\Ai Internship\House Price Dataset\Housing.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found at: {csv_path}")

df = pd.read_csv(csv_path)
print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

display(df.head())

print("\nInfo:")
print(df.info())

print("\nSummary Statistics (numeric columns):")
display(df.describe())

print("\nMissing values per column (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

# -------------------------------------------
# Step 2 — Identify Target Column (Price)
# -------------------------------------------

# Try to guess the target column name
possible_targets = ["SalePrice", "saleprice", "Price", "price", "MEDV", "medv"]
target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise KeyError(
        "❌ Could not find target column (SalePrice/Price/MEDV). "
        "Check df.columns and set target_col manually."
    )

print(f"\n🎯 Using target column: {target_col}")

# Drop rows where target is missing
df = df.dropna(subset=[target_col])

# -------------------------------------------
# Step 3 — Basic Cleaning (Drop ID-like columns if any)
# -------------------------------------------

id_like_cols = ["Id", "ID", "id"]
for col in id_like_cols:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"🔹 Dropped ID column: {col}")

# -------------------------------------------
# Step 4 — Separate Features and Target
# -------------------------------------------

y = df[target_col]
X = df.drop(columns=[target_col])

print("\nInitial feature shape:", X.shape)

# -------------------------------------------
# Step 5 — Handle Missing Values & Categorical Variables
# -------------------------------------------

numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

print("\nNumeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# Fill numeric NaNs with median
for col in numeric_cols:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Fill categorical NaNs with mode
for col in categorical_cols:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0])

print("\nMissing values AFTER filling (top 10):")
print(X.isna().sum().sort_values(ascending=False).head(10))

# One-hot encode categorical variables
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print("\n✅ One-hot encoding applied.")
else:
    print("\nℹ️ No categorical columns to encode.")

print("Final feature shape after encoding:", X.shape)

# -------------------------------------------
# Step 6 — Train/Test Split
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# -------------------------------------------
# Step 7 — Scaling for Linear Regression
# -------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------------------
# Step 8 — Train Models
# -------------------------------------------

# 8a. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# 8b. Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)  # tree-based, no scaling needed

# -------------------------------------------
# Step 9 — Evaluation Function
# -------------------------------------------

def evaluate_regression_model(name, model, X_tr, X_te, y_tr, y_te):
    print(f"\n==================== {name} ====================")
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    
    mae_tr = mean_absolute_error(y_tr, y_pred_tr)
    rmse_tr = mean_squared_error(y_tr, y_pred_tr, squared=False)
    mae_te = mean_absolute_error(y_te, y_pred_te)
    rmse_te = mean_squared_error(y_te, y_pred_te, squared=False)
    
    print(f"Train MAE:  {mae_tr:.2f}")
    print(f"Train RMSE: {rmse_tr:.2f}")
    print(f"Test MAE:   {mae_te:.2f}")
    print(f"Test RMSE:  {rmse_te:.2f}")
    
    # Plot predicted vs actual on test set
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, y_pred_te, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted - {name}")
    
    # Diagonal line
    min_val = min(y_te.min(), y_pred_te.min())
    max_val = max(y_te.max(), y_pred_te.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.grid(True)
    plt.show()

# -------------------------------------------
# Step 10 — Evaluate Both Models
# -------------------------------------------

evaluate_regression_model(
    "Linear Regression",
    lin_reg,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

evaluate_regression_model(
    "Gradient Boosting Regressor",
    gbr,
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n✅ TASK 6 COMPLETED SUCCESSFULLY")
print("Now add markdown cells explaining:")
print("- Dataset and target column")
print("- Cleaning and encoding")
print("- MAE/RMSE comparison between models")
print("- Interpretation of Actual vs Predicted plots")


# In[ ]:




