"""
Chicken Spoilage Categorization Model
======================================
VOC-based (Volatile Organic Compound) presence/absence classification.

Focus: Use only individual VOC presence/absence as features.
Model: Random Forest for interpretability and performance.
Goal: Identify which VOCs indicate spoilage vs freshness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, recall_score, 
    precision_score, f1_score, accuracy_score
)


class DataProcessor:
    """Load, clean, and create VOC presence/absence features from DataAI.csv"""
    
    def __init__(self, data_path="../data/raw_data/DataAI.csv"):
        self.data_path = data_path
        self.df = None
        self.df_raw = None
        self.df_processed = None
        
    def load_data(self):
        """Load raw CSV with encoding handling"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.data_path, encoding=encoding)
                self.df = self.df.dropna(axis=1, how='all')
                
                # Keep raw copy for VOC lookup
                self.df_raw = self.df.copy()
                self.df_raw.columns = self.df_raw.columns.str.lower().str.strip()
                
                print(f"[OK] Loaded raw data: {self.df.shape} (encoding: {encoding})")
                return self.df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read CSV")
    
    def clean_data(self):
        """Process raw VOC data and create classification labels"""
        df_clean = self.df.copy()
        df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        
        print(f"Raw records: {len(df_clean)}")
        
        # Filter by VOC relevance
        if 'revalence index' in df_clean.columns:
            df_clean = df_clean[df_clean['revalence index'] >= 80]
            print(f"After relevance filter (>=80): {len(df_clean)}")
        
        # Auto-classify by microbial load percentiles
        load_col = None
        for col in df_clean.columns:
            if 'microbial' in col.lower() and 'load' in col.lower():
                load_col = col
                break
        
        if load_col:
            p33, p67 = df_clean[load_col].quantile([0.33, 0.67])
            df_clean['class_label'] = pd.cut(
                df_clean[load_col],
                bins=[-np.inf, p33, p67, np.inf],
                labels=['fresh', 'moderate', 'spoiled']
            )
        
        # Aggregate by sample_id
        if 'sample_id' in df_clean.columns:
            df_agg = df_clean.groupby('sample_id').agg({
                'treatment': 'first',
                'day': 'first',
                'class_label': 'first'
            }).reset_index()
        else:
            df_agg = df_clean
        
        self.df_processed = df_agg
        print(f"[OK] Processed: {len(df_agg)} samples")
        return df_agg
    
    def get_features(self):
        """
        Create VOC presence/absence features.
        Only uses VOCs that strongly predict spoilage.
        """
        # Top spoilage-indicative VOCs (80-100% present in spoiled samples)
        spoilage_vocs = [
            '5-Methylundecane', '2-Methylheptane', 'Undecane', 'terpinolene',
            'Decane, 5-ethyl', 'Cyclopentasiloxane, decamethyl-', 'Decane, 4-ethyl',
            '6-Methylundecane', '4-Methylheptane', '2-Octanone', '3-hexanol',
            '2-methylbutanal'
        ]
        
        # Top fresh-indicative VOCs (85%+ present in fresh samples)
        fresh_vocs = [
            'Hexanal', '2-butanol', 'S(+)-2-butanol', 'butan-2-one',
            'Tridecane', 'butane-2,3-dione'
        ]
        
        indicator_vocs = spoilage_vocs + fresh_vocs
        
        # Create binary VOC presence features
        feature_data = []
        
        for sample_id in self.df_processed['sample_id'].unique():
            sample_row = self.df_processed[self.df_processed['sample_id'] == sample_id].iloc[0]
            features = {'sample_id': sample_id}
            
            # Check presence of each VOC in this sample
            for voc in indicator_vocs:
                voc_present = 1 if voc in self.df_raw[
                    self.df_raw['sample_id'] == sample_id
                ]['voc'].values else 0
                features[f'voc_{voc}'] = voc_present
            
            features['class_label'] = sample_row['class_label']
            feature_data.append(features)
        
        X_voc = pd.DataFrame(feature_data)
        y = X_voc['class_label']
        X = X_voc.drop(columns=['sample_id', 'class_label'])
        
        print(f"\n[OK] VOC Features Created:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]} (VOC presence/absence)")
        print(f"  Spoilage VOCs: {len(spoilage_vocs)}")
        print(f"  Fresh VOCs: {len(fresh_vocs)}")
        
        return X, y


class CategorizationModel:
    """Train and evaluate Random Forest on VOC features"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.evaluation_results = {}
        self.predictions = {}
        self.feature_names = None
        self.optimal_threshold = 0.5
        
    def prepare_data(self, X, y):
        """Split and scale data (70-15-15)"""
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        # Second split: 50-50 of temp (15% val, 15% test)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        # Scale features
        numeric_cols = X_train.columns.tolist()
        
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
            'y_orig': y_train
        }
        self.val_data = {
            'X': X_val_scaled, 'y': y_val_encoded,
            'y_orig': y_val
        }
        self.test_data = {
            'X': X_test_scaled, 'y': y_test_encoded,
            'y_orig': y_test
        }
        self.feature_names = X_train_scaled.columns.tolist()
        
        print(f"[OK] Data split: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
        return self.train_data, self.val_data, self.test_data
    
    def train(self):
        """Train regularized Random Forest"""
        X_train, y_train = self.train_data['X'], self.train_data['y']
        
        # Regularized RF to prevent overfitting on small dataset
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Shallow trees
            min_samples_leaf=3,  # Prevent leaf memorization
            min_samples_split=5,  # Prevent over-splitting
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Store predictions for visualization
        self.predictions = {
            'train': self.model.predict(self.train_data['X']),
            'val': self.model.predict(self.val_data['X']),
            'test': self.model.predict(self.test_data['X']),
            'train_proba': self.model.predict_proba(self.train_data['X']),
            'val_proba': self.model.predict_proba(self.val_data['X']),
            'test_proba': self.model.predict_proba(self.test_data['X'])
        }
        
        print(f"[OK] Random Forest trained")
    
    def evaluate(self):
        """Evaluate on test set"""
        y_test = self.test_data['y']
        y_test_pred = self.model.predict(self.test_data['X'])
        
        cm = confusion_matrix(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
        recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
        accuracy = accuracy_score(y_test, y_test_pred)
        
        self.evaluation_results = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'recall_spoiled': recall[-1]  # Last class = spoiled
        }
        
        print("\n" + "="*70)
        print("MODEL EVALUATION (TEST SET)")
        print("="*70)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall (Spoilage): {recall[-1]:.4f}")
        print(f"Precision (Avg): {precision.mean():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=self.label_encoder.classes_,
                                  digits=3))
    
    def optimize_threshold(self, target_recall=0.95):
        """Find optimal threshold for high spoilage recall"""
        val_proba = self.model.predict_proba(self.val_data['X'])
        y_val = self.val_data['y']
        
        spoiled_class_idx = len(self.label_encoder.classes_) - 1
        
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
        print(f"\n[OK] Optimal threshold: {best_threshold:.2f} (recall: {best_recall:.4f})")
    
    def get_feature_importance(self):
        """Return top predictive VOCs"""
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'voc': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, output_dir='../model_pkls'):
        """Save trained model"""
        Path(output_dir).mkdir(exist_ok=True)
        
        metadata = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        joblib.dump(metadata, f'{output_dir}/categorization_model.pkl')
        print(f"[OK] Model saved")

    
    def clean_data(self):
        """
        Process raw VOC data:
          1. Standardize column names
          2. Filter VOCs by Revalence Index >= 80
          3. Auto-classify samples by microbial load percentiles
          4. Aggregate features per sample
        """
        df_clean = self.df.copy()
        
        # Remove empty/unnamed columns
        df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
        
        # Standardize column names (lowercase, strip spaces)
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        
        print(f"Raw records before filtering: {len(df_clean)}")
        print(f"Columns available: {df_clean.columns.tolist()}")
        
        # Filter by VOC relevance (keep high-quality VOCs)
        if 'revalence index' in df_clean.columns:
            df_clean = df_clean[df_clean['revalence index'] >= 80]
            print(f"Records after relevance filter (>=80): {len(df_clean)}")
        
        # Auto-classify based on microbial load percentiles
        load_col = None
        for col in df_clean.columns:
            if 'microbial' in col.lower() and 'load' in col.lower():
                load_col = col
                break
        
        if load_col:
            p33, p67 = df_clean[load_col].quantile([0.33, 0.67])
            
            df_clean['class_label'] = pd.cut(
                df_clean[load_col],
                bins=[-np.inf, p33, p67, np.inf],
                labels=['fresh', 'moderate', 'spoiled']
            )
            print(f"Classes created from {load_col} (33rd: {p33:.2f}, 67th: {p67:.2f})")
        else:
            df_clean['class_label'] = 'fresh'
            print("Warning: No microbial load column found, using default 'fresh' label")
        
        # Aggregate by sample (group by sample_id)
        group_cols = [c for c in ['sample_id', 'treatment', 'day', 'replicate '] 
                     if c in df_clean.columns]
        
        if 'sample_id' in df_clean.columns:
            # Group by sample_id primarily
            agg_dict = {
                'treatment': 'first',
                'day': 'first',
                'revalence index': 'mean',
                'voc': 'count',
                'class_label': 'first'
            }
            df_agg = df_clean.groupby('sample_id').agg(agg_dict).reset_index()
        elif group_cols:
            # Fall back to grouping by treatment+day+replicate
            agg_dict = {c: 'first' for c in group_cols if c != 'sample_id'}
            agg_dict['revalence index'] = 'mean'
            agg_dict['voc'] = 'count'
            agg_dict['class_label'] = 'first'
            df_agg = df_clean.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            df_agg = df_clean
        
        # Rename voc count column
        if 'voc' in df_agg.columns:
            df_agg.rename(columns={'voc': 'voc_count'}, inplace=True)
        
        # Standardize column names again for consistency
        df_agg.columns = df_agg.columns.str.lower().str.strip()
        
        # Create VOC diversity ratio (normalized voc_count)
        if 'voc_count' in df_agg.columns:
            voc_min, voc_max = df_agg['voc_count'].min(), df_agg['voc_count'].max()
            if voc_max > voc_min:
                df_agg['voc_diversity_ratio'] = (df_agg['voc_count'] - voc_min) / (voc_max - voc_min)
            else:
                df_agg['voc_diversity_ratio'] = 0.5
        
        self.df_processed = df_agg
        print(f"[OK] Processed data: {self.df_processed.shape[0]} samples x {self.df_processed.shape[1]} features")
        print(f"Features: {self.df_processed.columns.tolist()}")
        return self.df_processed
    
    def get_features(self):
        """
        Extract features based on INDIVIDUAL VOC PRESENCE, not aggregation.
        
        Strategy:
          1. Identify top spoilage indicator VOCs (80-100% in spoiled samples)
          2. Create binary features for each VOC: present (1) or absent (0) per sample
          3. This prevents overfitting on 'day' and focuses on chemical signatures
        """
        # Identify top spoilage-indicative VOCs from full dataset
        # These VOCs appear predominantly in spoiled samples
        spoilage_vocs = [
            '5-Methylundecane', '2-Methylheptane', 'Undecane', 'terpinolene',
            'Decane, 5-ethyl', 'Cyclopentasiloxane, decamethyl-', 'Decane, 4-ethyl',
            '6-Methylundecane', '4-Methylheptane', '2-Octanone', '3-hexanol',
            '2-methylbutanal', 'Propyl 2-methyl-2-propenoate'
        ]
        
        # Identify top fresh-indicative VOCs
        fresh_vocs = [
            'Hexanal', '2-butanol', 'S(+)-2-butanol', 'Tridecane',
            'butan-2-one', 'Butyl methacrylate', 'butane-2,3-dione',
            'Methane, bromochloro-', '2-nonanol', '1-methoxy-2-propanol'
        ]
        
        indicator_vocs = spoilage_vocs + fresh_vocs
        
        # Create binary VOC presence features for each sample
        feature_data = []
        
        for sample_id in self.df_processed['sample_id'].unique():
            sample_row = self.df_processed[self.df_processed['sample_id'] == sample_id].iloc[0]
            features = {'sample_id': sample_id}
            
            # Add treatment
            features['treatment'] = sample_row['treatment']
            
            # For each indicator VOC, check if present in this sample
            for voc in indicator_vocs:
                # Check if this VOC appears in the original data for this sample
                voc_present = 1 if voc in self.df_raw[
                    self.df_raw['sample_id'] == sample_id
                ]['voc'].values else 0
                features[f'voc_{voc}'] = voc_present
            
            # Add target
            features['class_label'] = sample_row['class_label']
            
            feature_data.append(features)
        
        X_voc = pd.DataFrame(feature_data)
        y = X_voc['class_label']
        
        # Remove non-feature columns
        X = X_voc.drop(columns=['sample_id', 'class_label'])
        
        print(f"\n[OK] VOC-Based Features Created:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Spoilage VOC indicators: {len(spoilage_vocs)}")
        print(f"  Fresh VOC indicators: {len(fresh_vocs)}")
        print(f"\nFeature columns (VOC presence/absence):")
        print(f"  {X.columns.tolist()}")
        
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
        
        print(f"[OK] Data prepared: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
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
        
        # Random Forest - REGULARIZED to prevent overfitting on small dataset
        # With 35 training samples and 23 features, we need strong regularization
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,  # REDUCED from 10 to 5 - prevent deep memorization
            min_samples_leaf=3,  # ADDED - require at least 3 samples per leaf
            min_samples_split=5,  # ADDED - require 5 samples to split node
            class_weight='balanced', 
            random_state=self.random_state, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           objective='multi:softmax', num_class=3, random_state=self.random_state, n_jobs=-1)
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = xgb
        
        print(f"[OK] Models trained: {list(self.models.keys())}")
    
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
        print(f"\n[OK] Model saved to {output_dir}/categorization_model.pkl")
