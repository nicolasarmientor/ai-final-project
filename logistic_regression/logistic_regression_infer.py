import pandas as pd
import joblib

# Load model
model = joblib.load("model_pkls/logistic_regression_model.pkl")

print("\nEnter sample information for spoilage prediction:\n")

treatment = input("Treatment (e.g., Control, TA 1, TA 2, ...): ").strip()
day = int(input("Day (e.g. 0, 1, 2, ...): "))
revalence_index = float(input("Revalence index (e.g. 95.44): "))
voc_count = int(input("VOC count (e.g. 33): "))

data = {
    "treatment": [treatment],
    "day": [day],
    "revalence_index": [revalence_index],
    "voc_count": [voc_count]
}

X_new = pd.DataFrame(data)

# Predict class and probabilities
pred_class = model.predict(X_new)[0]
pred_prob = model.predict_proba(X_new)[0]

classes = model.named_steps["clf"].classes_

print("\nPredicted class:", pred_class)
print("\nClass probabilities:")
for cls, p in zip(classes, pred_prob):
    print(f"  {cls}: {p:.3f}")