import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================
# Create Required Directories
# ============================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================
# Load Dataset
# ============================

try:
    df = pd.read_csv("House_Price_Dataset.csv")
    print("Dataset Loaded Successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# ============================
# Display Dataset Info
# ============================

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# ============================
# Data Cleaning
# ============================

df = df.dropna()

# ============================
# Features & Target
# ============================

FEATURES = ['Area', 'Bedrooms', 'Bathrooms']
TARGET = 'Price'

# ============================
# Correlation Heatmap
# ============================

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# ============================
# Feature Selection
# ============================

X = df[FEATURES]
y = df[TARGET]

# ============================
# Train Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================
# Model Training
# ============================

model = LinearRegression()

model.fit(X_train, y_train)

# ============================
# Prediction
# ============================

y_pred = model.predict(X_test)

# ============================
# Evaluation
# ============================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# ============================
# Save Metrics
# ============================

with open("outputs/model_metrics.txt", "w") as f:
    f.write("House Price Prediction Metrics\n")
    f.write("=" * 40 + "\n")
    f.write(f"MAE : {mae:.2f}\n")
    f.write(f"MSE : {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

# ============================
# Actual vs Predicted Plot
# ============================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# ============================
# Save Model
# ============================

joblib.dump(
    model,
    "models/house_price_model.pkl"
)

print("\nModel Saved Successfully")

# ============================
# Sample Prediction
# ============================

sample_house = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Bathrooms": [2]
})

prediction = model.predict(sample_house)

print("\nSample House Prediction")
print(f"Predicted Price: ₹{prediction[0]:,.2f}")

print("\nProject Completed Successfully")