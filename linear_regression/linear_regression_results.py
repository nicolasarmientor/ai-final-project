import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load model
model = joblib.load("model_pkls/linear_regression_model.pkl")

# Load data
df = pd.read_csv("data/processed_data/linear_regression_data.csv")

# Features and target
X = df[["day", "revalence_index"]]
y = df["microbial_load"]

# Microbial load prediction
y_pred = model.predict(X)
df["predicted_microbial_load"] = y_pred

# Metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print("\n### Linear Regression Evaluation (Full Dataset) ###")
print(f"R^2:   {r2:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}\n")

# Plot
coef = np.polyfit(df["day"], df["predicted_microbial_load"], deg=1)
poly_line = np.poly1d(coef)

day_line = np.linspace(df["day"].min(), df["day"].max(), 200)
pred_line = poly_line(day_line)

plt.figure(figsize=(10, 6))
plt.scatter(df["day"], df["microbial_load"], label="Actual", alpha=0.7)
plt.scatter(df["day"], df["predicted_microbial_load"], label="Predicted", alpha=0.7)
plt.plot(day_line, pred_line, color="red", linewidth=2, label="Best-Fit Line")

plt.xlabel("Day")
plt.ylabel("Microbial Load")
plt.title("Actual & Predicted Microbial Load vs Day")

plt.legend()
plt.tight_layout()
plt.savefig("linear_regression/figure/linear_regression_plot.png")

plt.show()