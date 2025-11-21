import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
df = pd.read_csv("data/processed_data/linear_regression_data.csv")

# Features and target
X = df[["day", "revalence_index"]]
y = df["microbial_load"]

# 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3, 
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    random_state=42, 
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Save model
joblib.dump(model, "model_pkls/linear_regression_model.pkl")