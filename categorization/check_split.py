import pandas as pd
import numpy as np
from categorization import DataProcessor, CategorizationModel
from sklearn.model_selection import train_test_split

# Load and process data
processor = DataProcessor('../data/raw_data/DataAI.csv')
processor.load_data()
processor.clean_data()
X, y = processor.get_features()

# Replicate the exact split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Check spoiled distribution
print('VAL SET:')
print(f'  Total: {len(y_val)}')
print(f'  Spoiled: {(y_val == "spoiled").sum()}')
print(f'  Distribution: {y_val.value_counts().to_dict()}')

print('\nTEST SET:')
print(f'  Total: {len(y_test)}')
print(f'  Spoiled: {(y_test == "spoiled").sum()}')
print(f'  Distribution: {y_test.value_counts().to_dict()}')

print("\n" + "="*50)
print("ISSUE: The threshold is optimized on VALIDATION set")
print("but evaluated on TEST set with different spoilage distribution!")
print("="*50)
