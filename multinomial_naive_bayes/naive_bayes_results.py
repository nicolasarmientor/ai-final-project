import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed_data/naive_bayes_data.csv")

df["voc"] = df["voc"].apply(ast.literal_eval)

X_list = df["voc"]
y = df["class_label"]

# Load model and mlb
model = joblib.load("model_pkls/naive_bayes_model.pkl")
mlb = joblib.load("model_pkls/naive_bayes_mlb.pkl")

X = mlb.transform(X_list)

# 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3, 
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.9,
    random_state=42,
    stratify=y_temp
)

# Test metrics
y_test_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

c_matrix = confusion_matrix(y_test, y_test_pred)
labels = ["fresh", "moderate", "spoiled"]

plt.figure(figsize=(7, 5))
sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()

plt.savefig("multinomial_naive_bayes/figure/nb_confusion_matrix_heatmap.png")
plt.show()
