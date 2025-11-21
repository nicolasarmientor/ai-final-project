import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("model_pkls/linear_regression_model.pkl")

# User input
print("\nEnter values to predict microbial load")
day = float(input("Day: "))
revalence_index = float(input("Revalence index: "))

# Predict microbial load
X_input = np.array([[day, revalence_index]])
pred = model.predict(X_input)[0]

print(f"\nPredicted microbial load: {pred:.2f}\n")