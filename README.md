# House-Price-Prediction-System
📌 Objective

The objective of this task is to build a machine learning model that predicts house prices based on various numerical and categorical property features. This task demonstrates practical regression modeling, data cleaning, feature engineering, and evaluation using real-world housing data.

📂 Dataset

File: Housing.csv

Location: Provided local dataset (House Price Dataset)

Contains a mix of:

Numerical features (size, area, age, floors, etc.)

Categorical features (location, house type, quality, etc.)

Target Column: Price

 Steps Performed
1. Dataset Loading

Loaded dataset using pandas

Displayed:

Shape (rows × columns)

Columns list

First few rows

.info()

.describe() statistics

Missing values per column

2. Data Cleaning & Preprocessing

Handled:

Missing numeric values → filled with median

Missing categorical values → filled with mode

Dropped ID-like columns (Id, ID, etc.) if present

Converted categorical features to numeric using One-Hot Encoding

This ensures the dataset is ready for machine learning models.

3. Feature Engineering

Identified numerical and categorical columns

Applied scaling (StandardScaler) for Linear Regression

Left unscaled data for tree-based models (GBR)

Split dataset into training (80%) and testing (20%)

4. Machine Learning Models Developed

Two ML regression algorithms were trained to predict house prices:

Model	Type	Notes
Linear Regression	Basic ML Algorithm	Good baseline, requires scaling
Gradient Boosting Regressor (GBR)	Ensemble ML Algorithm	Handles non-linearity, generally more accurate
5. Model Evaluation Metrics

Each model was evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

Train vs Test performance comparison

6. Visualization

Generated:

Scatter plot: Actual vs Predicted Prices

Diagonal reference line to measure accuracy visually

Helps understand underprediction/overprediction

 Key Insights

GBR significantly outperforms Linear Regression in most real-estate datasets due to:

Non-linear pattern detection

Handling of categorical interactions

Robustness to outliers

Features like living area, number of rooms, overall quality, and location typically influence prices the most.

Proper encoding and preprocessing greatly improved model performance.
