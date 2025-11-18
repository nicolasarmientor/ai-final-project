from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import joblib

df = pd.read_csv("data/processed_data/naive_bayes_data.csv")

df["voc"] = df["voc"].apply(ast.literal_eval)

# Set target and features
X_list = df["voc"]
y = df["class_label"]

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(X_list)

# 70/15/15 split
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

model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, "model_pkls/naive_bayes_model.pkl")
joblib.dump(mlb, "model_pkls/naive_bayes_mlb.pkl")
