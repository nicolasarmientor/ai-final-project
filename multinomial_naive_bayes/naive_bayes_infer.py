import pandas as pd
import joblib

def clean_list(voc_input: str):
    return [v.strip().lower() for v in voc_input.split(",") if v.strip() != ""]


# Load model
model = joblib.load("model_pkls/naive_bayes_model.pkl")
mlb = joblib.load("model_pkls/naive_bayes_mlb.pkl")

classes = model.classes_

print("\nEnter VOC compounds separated by commas")
print("Example: ethanol, hexanal, 2-butanone, ...")

voc_input = input("\nVOCs: ")
voc_list = clean_list(voc_input)

X_new = mlb.transform([voc_list])
pred_class = model.predict(X_new)[0]
pred_prob = model.predict_proba(X_new)[0]

print("\nPredicted class:", pred_class)
print("\nClass probabilities:")
for cls, p in zip(classes, pred_prob):
    print(f"  {cls}: {p:.3f}")
print("\n")
