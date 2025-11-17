import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load logistic regression dataset
df = pd.read_csv("data/processed_data/logistic_regression_data.csv")

# Feature set and target
X = df.drop(columns=["class_label"])
y = df["class_label"]

# Columns
numeric_features = ["day", "revalence_index", "voc_count"]
categorical_features = ["treatment"]

# Preprocessing pipeline
numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Logistic regression model
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        max_iter=500, 
        solver="lbfgs"
    ))
])

# 70/15/15 train/test/val split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3, 
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    random_state=42, 
    stratify=y_temp
)

# Train model
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

y_test_pred = model.predict(X_test)

# Save model a pkl
joblib.dump(model, "model_pkls/logistic_regression_model.pkl")

print("\nModel saved successfully!\n")