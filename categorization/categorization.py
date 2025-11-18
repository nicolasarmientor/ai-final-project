"""
Chicken Spoilage Categorization Model
======================================
Hybrid approach combining VOC aggregation with advanced ML for freshness classification.

Data Flow:
  Raw DataAI.csv → Filter VOCs (≥80 relevance) → Auto-classify by microbial load percentiles
  → Aggregate per sample (5 features) → Train LR/RF/XGBoost → Optimize for 95%+ recall
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, recall_score, 
    precision_score, f1_score, accuracy_score
)


class DataProcessor:
    """Load, clean, and preprocess raw VOC data from DataAI.csv"""
    
    def __init__(self, data_path="../data/raw_data/DataAI.csv"):
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        
    def load_data(self):
        """Load and remove empty columns from raw CSV"""
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna(axis=1, how='all')  # Remove empty columns
        print(f"✓ Loaded raw data: {self.df.shape}")
        return self.df
    
    def clean_data(self):
        """
        Process raw VOC data:
          1. Standardize column names
          2. Filter VOCs by revalence_index ≥ 80
          3. Auto-classify samples by microbial load percentiles
          4. Aggregate features per sample
        """
        df_clean = self.df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        
        print(f"Raw records before filtering: {len(df_clean)}")
        
        # Filter by VOC relevance (keep high-quality VOCs)
        if 'revalence_index' in df_clean.columns:
            df_clean = df_clean[df_clean['revalence_index'] >= 80]
            print(f"Records after relevance filter (≥80): {len(df_clean)}")
        
        # Auto-classify based on microbial load percentiles
        if 'microbial_load' in df_clean.columns or 'microbial_load (log)' in df_clean.columns:
            load_col = 'microbial_load' if 'microbial_load' in df_clean.columns else 'microbial_load (log)'
            p33, p67 = df_clean[load_col].quantile([0.33, 0.67])
            
            df_clean['class_label'] = pd.cut(
                df_clean[load_col],
                bins=[-np.inf, p33, p67, np.inf],
                labels=['fresh', 'moderate', 'spoiled']
            )
        else:
            # Fallback: create dummy labels
            df_clean['class_label'] = 'fresh'
        
        # Aggregate by sample (group by sample_id or row identifier)
        if 'sample_id' in df_clean.columns:
            agg_dict = {
                'treatment': 'first',
                'day': 'first',
                'revalence_index': 'mean',
                'voc': 'count',  # Becomes voc_count
                'class_label': 'first'
            }
            df_agg = df_clean.groupby('sample_id').agg(agg_dict).reset_index()
        else:
            # If no sample_id, aggregate by combination of identifying columns
            group_cols = [c for c in ['treatment', 'day', 'replicate'] if c in df_clean.columns]
            if group_cols:
                agg_dict = {c: 'first' for c in group_cols}
                agg_dict['revalence_index'] = 'mean'
                agg_dict['voc'] = 'count'
                agg_dict['class_label'] = 'first'
                df_agg = df_clean.groupby(group_cols).agg(agg_dict).reset_index()
            else:
                df_agg = df_clean
        
        # Rename voc count column
        if 'voc' in df_agg.columns:
            df_agg.rename(columns={'voc': 'voc_count'}, inplace=True)
        
        # Create VOC diversity ratio (normalized voc_count)
        if 'voc_count' in df_agg.columns:
            voc_min, voc_max = df_agg['voc_count'].min(), df_agg['voc_count'].max()
            if voc_max > voc_min:
                df_agg['voc_diversity_ratio'] = (df_agg['voc_count'] - voc_min) / (voc_max - voc_min)
            else:
                df_agg['voc_diversity_ratio'] = 0
        
        self.df_processed = df_agg
        print(f"✓ Processed data: {self.df_processed.shape[0]} samples × {self.df_processed.shape[1]} features")
        return self.df_processed
    
    def get_features(self):
        """Extract feature matrix and target vector"""
        feature_cols = ['day', 'revalence_index', 'voc_count', 'voc_diversity_ratio', 'treatment']
        available_cols = [c for c in feature_cols if c in self.df_processed.columns]
        
        X = self.df_processed[available_cols]
        y = self.df_processed['class_label']
        
        return X, y


class CategorizationModels:
    """Train and evaluate classification models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.evaluation_results = {}
        
    def prepare_data(self, X, y):
        """Encode, scale, and split data"""
        # Encode categorical features (treatment)
        X_encoded = pd.get_dummies(X, columns=['treatment'], drop_first=True)
        
        # Train-test-validation split (70-15-15)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        # Ensure all splits have same columns
        for split in [X_val, X_test]:
            for col in X_train.columns:
                if col not in split.columns:
                    split[col] = 0
            for col in split.columns:
                if col not in X_train.columns:
                    split.drop(col, axis=1, inplace=True)
            split = split[X_train.columns]
        
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        
        # Scale numeric features
        numeric_cols = [c for c in X_train.columns if c != 'treatment' and not c.startswith('treatment_')]
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        X_val_scaled = X_val.copy()
        X_val_scaled[numeric_cols] = self.scaler.transform(X_val[numeric_cols])
        
        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        # Encode target
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        self.train_data = {
            'X': X_train_scaled, 'y': y_train_encoded,
            'X_orig': X_train, 'y_orig': y_train
        }
        self.val_data = {
            'X': X_val_scaled, 'y': y_val_encoded,
            'X_orig': X_val, 'y_orig': y_val
        }
        self.test_data = {
            'X': X_test_scaled, 'y': y_test_encoded,
            'X_orig': X_test, 'y_orig': y_test
        }
        self.feature_names = X_train_scaled.columns.tolist()
        
        print(f"✓ Data prepared: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
        return self.train_data, self.val_data, self.test_data
    
    def train_models(self):
        """Train Logistic Regression, Random Forest, and XGBoost"""
        X_train, y_train = self.train_data['X'], self.train_data['y']
        
        print("\nTraining models...")
        
        # Logistic Regression
        lr = LogisticRegression(multi_class='multinomial', class_weight='balanced', 
                               max_iter=500, random_state=self.random_state)
        lr.fit(X_train, y_train)
        self.models['LogisticRegression'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   class_weight='balanced', random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           objective='multi:softmax', num_class=3, random_state=self.random_state, n_jobs=-1)
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = xgb
        
        print(f"✓ Models trained: {list(self.models.keys())}")
    
    def get_predictions(self):
        """Generate predictions for all models on all sets"""
        for model_name, model in self.models.items():
            self.predictions[model_name] = {
                'train': model.predict(self.train_data['X']),
                'val': model.predict(self.val_data['X']),
                'test': model.predict(self.test_data['X']),
                'train_proba': model.predict_proba(self.train_data['X']),
                'val_proba': model.predict_proba(self.val_data['X']),
                'test_proba': model.predict_proba(self.test_data['X'])
            }
    
    def evaluate(self):
        """Evaluate models on test set"""
        y_test = self.test_data['y']
        
        for model_name in self.models.keys():
            y_pred = self.predictions[model_name]['test']
            
            cm = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.evaluation_results[model_name] = {
                'confusion_matrix': cm,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'recall_spoiled': recall[-1]  # Class 2 (last) = spoiled
            }
        
        print("\n" + "="*70)
        print("MODEL EVALUATION (TEST SET)")
        print("="*70)
        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Recall (Spoiled): {results['recall_spoiled']:.4f}")
            print(f"  Precision (Avg): {results['precision'].mean():.4f}")
    
    def optimize_threshold(self, target_recall=0.95):
        """Find optimal decision threshold for high spoilage recall"""
        print(f"\n{'='*70}")
        print(f"THRESHOLD OPTIMIZATION (Target Recall: {target_recall:.2f})")
        print("="*70)
        
        # Use Random Forest for threshold optimization
        model_name = 'RandomForest'
        val_proba = self.predictions[model_name]['val_proba']
        y_val = self.val_data['y']
        
        spoiled_class_idx = len(self.label_encoder.classes_) - 1  # Last class = spoiled
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_recall = 0
        
        for threshold in thresholds:
            y_pred = np.argmax(val_proba, axis=1)
            for i in range(len(val_proba)):
                if val_proba[i, spoiled_class_idx] > threshold:
                    y_pred[i] = spoiled_class_idx
            
            recall = recall_score(y_val, y_pred, average=None, zero_division=0)[spoiled_class_idx]
            
            if recall >= target_recall and recall > best_recall:
                best_recall = recall
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        print(f"\nOptimal threshold: {best_threshold:.2f}")
        print(f"Expected recall on validation: {best_recall:.4f}")
        
        # Apply to test set
        test_proba = self.predictions[model_name]['test_proba']
        y_test = self.test_data['y']
        
        y_pred_optimized = np.argmax(test_proba, axis=1)
        for i in range(len(test_proba)):
            if test_proba[i, spoiled_class_idx] > best_threshold:
                y_pred_optimized[i] = spoiled_class_idx
        
        recall_optimized = recall_score(y_test, y_pred_optimized, average=None, zero_division=0)[spoiled_class_idx]
        precision_optimized = precision_score(y_test, y_pred_optimized, average=None, zero_division=0)[spoiled_class_idx]
        
        self.evaluation_results[f'{model_name} (Optimized)'] = {
            'confusion_matrix': confusion_matrix(y_test, y_pred_optimized),
            'recall_spoiled': recall_optimized,
            'precision_spoiled': precision_optimized,
            'y_pred': y_pred_optimized
        }
        
        print(f"\nTest Set Results (Optimized):")
        print(f"  Recall (Spoiled): {recall_optimized:.4f}")
        print(f"  Precision (Spoiled): {precision_optimized:.4f}")
    
    def save_best_model(self, output_dir='model_pkls'):
        """Save Random Forest model with metadata"""
        Path(output_dir).mkdir(exist_ok=True)
        
        model = self.models['RandomForest']
        metadata = {
            'model': model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        joblib.dump(metadata, f'{output_dir}/categorization_model.pkl')
        print(f"\n✓ Model saved to {output_dir}/categorization_model.pkl")
