import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from logistic_regression_train import y_test, y_test_pred
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load model
model = joblib.load("model_pkls/logistic_regression_model.pkl")

# Load processed data
df = pd.read_csv("data/processed_data/logistic_regression_data.csv")

# Separate features and labels
X = df.drop(columns=["class_label"])
y = df["class_label"]

# Recreate 70/15/15 split
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

y_test_pred = model.predict(X_test)

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Create confusion matrix heatmap
c_matrix = confusion_matrix(y_test, y_test_pred)
labels = ["fresh", "moderate", "spoiled"]

plt.figure(figsize=(7, 5))
sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()

plt.savefig("logistic_regression/figure/lr_confusion_matrix_heatmap.png")
plt.show()