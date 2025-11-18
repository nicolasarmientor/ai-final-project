"""
Model Training Module
Train and compare multiple classification models.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


class ModelTrainer:
    """Train and manage multiple models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.predictions = {}
        self.history = {}
    
    def split_data(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Split data into train, validation, and test sets (70-15-15).
        
        Args:
            X: Features
            y: Target
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "=" * 80)
        print("TRAIN-VALIDATION-TEST SPLIT (70-15-15)")
        print("=" * 80)
        
        # First split: 70% train / 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.3,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: split temp 50-50 into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\nSplit Results:")
        print(f"  Training Set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation Set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test Set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Verify stratification
        print(f"\nClass Distribution Across Splits:")
        for split_name, y_split in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
            print(f"\n{split_name}:")
            print(f"  {dict(y_split.value_counts())}")
        
        print(f"\n✓ Data split complete with stratification!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model."""
        print("\n" + "=" * 80)
        print("[1/3] TRAINING LOGISTIC REGRESSION")
        print("=" * 80)
        
        model = LogisticRegression(
            multi_class='multinomial',
            class_weight='balanced',
            max_iter=500,
            solver='lbfgs',
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        
        print("\n✓ Logistic Regression trained successfully")
        print(f"  Classes: {model.classes_}")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model."""
        print("\n" + "=" * 80)
        print("[2/3] TRAINING RANDOM FOREST")
        print("=" * 80)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        
        print("\n✓ Random Forest trained successfully")
        print(f"  N Estimators: 100")
        print(f"  Max Depth: 10")
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        print("\n" + "=" * 80)
        print("[3/3] TRAINING XGBOOST")
        print("=" * 80)
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train != 2).sum()  # Not spoiled
        pos_count = (y_train == 2).sum()  # Spoiled
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=3,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        
        print("\n✓ XGBoost trained successfully")
        print(f"  N Estimators: 100")
        print(f"  Max Depth: 6")
        print(f"  Scale POS Weight: {scale_pos_weight:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all models."""
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        print("\n" + "=" * 80)
        print("✓ ALL MODELS TRAINED!")
        print("=" * 80)
        
        return self.models
    
    def predict_all(self, X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded):
        """Make predictions on all sets for all models."""
        print("\n" + "=" * 80)
        print("GENERATING PREDICTIONS")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            train_proba = model.predict_proba(X_train)
            val_proba = model.predict_proba(X_val)
            test_proba = model.predict_proba(X_test)
            
            self.predictions[model_name] = {
                'train': train_pred, 'val': val_pred, 'test': test_pred,
                'train_proba': train_proba, 'val_proba': val_proba, 'test_proba': test_proba
            }
            
            print(f"  ✓ Predictions generated")
        
        print("\n✓ Predictions complete!")
        return self.predictions
    
    def get_model(self, model_name):
        """Get trained model by name."""
        return self.models.get(model_name)
    
    def get_predictions(self, model_name):
        """Get predictions for model."""
        return self.predictions.get(model_name)
    
    def save_models(self, output_dir="model_pkls"):
        """Save all models."""
        print(f"\n✓ Saving models to {output_dir}/")
        for model_name, model in self.models.items():
            filename = f"{output_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"  ✓ {model_name} saved")
