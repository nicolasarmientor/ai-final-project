"""
Extended Validation Module
==========================
Test R1-trained model on R2 data without affecting R1 model
Provides separate validation and testing with R1+R2 combined
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, recall_score, 
    precision_score, f1_score, accuracy_score
)


class ExtendedValidator:
    """
    Validate R1-trained model on R2 data
    Keeps R1 training intact, only extends test/val sets
    """
    
    def __init__(self, model_path=None):
        """Load pre-trained R1 model"""
        if model_path is None:
            # Model is at project root: ../../../model_pkls/
            model_path = Path(__file__).parent.parent.parent / 'model_pkls' / 'categorization_model.pkl'
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.metadata = joblib.load(self.model_path)
        
        self.model = self.metadata['model']
        self.scaler = self.metadata['scaler']
        self.label_encoder = self.metadata['label_encoder']
        self.feature_names = self.metadata['feature_names']
        
        self.spoilage_vocs = [
            '5-Methylundecane', '2-Methylheptane', 'Undecane', 'terpinolene',
            'Decane, 5-ethyl', 'Cyclopentasiloxane, decamethyl-', 'Decane, 4-ethyl',
            '6-Methylundecane', '4-Methylheptane', '2-Octanone', '3-hexanol',
            '2-methylbutanal'
        ]
        
        self.fresh_vocs = [
            'Hexanal', '2-butanol', 'S(+)-2-butanol', 'butan-2-one',
            'Tridecane', 'butane-2,3-dione'
        ]
        
        self.indicator_vocs = self.spoilage_vocs + self.fresh_vocs
        
        print("[OK] Loaded R1-trained model")
        print(f"     Features: {len(self.feature_names)} VOCs")
        print(f"     Classes: {self.label_encoder.classes_}")
    
    def load_replicate_data(self, rep_name='R2', data_path=None):
        """Load R2 (or R3) raw data and create features"""
        
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / 'data' / 'raw_data'
        
        if rep_name == 'R1':
            file_path = data_path / 'DataAI.csv'
        else:
            file_path = data_path / f'DataAI {rep_name}.csv'
        
        df = pd.read_csv(file_path, encoding='latin-1')
        df.columns = df.columns.str.lower().str.strip()
        df = df.dropna(axis=1, how='all')
        
        # Handle numeric conversion
        for col in ['revalence index', 'day', 'microbial load (log)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter and classify
        df = df[df['revalence index'] >= 80]
        load_col = 'microbial load (log)'
        p33, p67 = df[load_col].quantile([0.33, 0.67])
        df['class_label'] = pd.cut(df[load_col], bins=[-np.inf, p33, p67, np.inf],
                                    labels=['fresh', 'moderate', 'spoiled'])
        
        print(f"\n[OK] Loaded {rep_name} raw data: {len(df)} records")
        
        # Aggregate to sample level
        samples_data = []
        for sample_id in df['sample_id'].dropna().unique():
            sample_rows = df[df['sample_id'] == sample_id]
            sample_data = {
                'sample_id': sample_id,
                'treatment': sample_rows['treatment'].iloc[0],
                'day': sample_rows['day'].iloc[0],
                'class_label': sample_rows['class_label'].iloc[0],
                'microbial_load': sample_rows[load_col].iloc[0],
                'replicate': rep_name
            }
            samples_data.append(sample_data)
        
        df_samples = pd.DataFrame(samples_data)
        print(f"[OK] Aggregated to: {len(df_samples)} samples")
        print(f"     Class distribution:\n{df_samples['class_label'].value_counts()}")
        
        # Create VOC features
        feature_data = []
        for idx, sample_row in df_samples.iterrows():
            sample_id = sample_row['sample_id']
            treatment = sample_row['treatment']
            
            # Get all VOCs for this sample across all rows
            sample_vocs = df[
                (df['sample_id'] == sample_id) & 
                (df['treatment'] == treatment)
            ]['voc'].values
            
            features = {'sample_id': sample_id, 'treatment': treatment}
            
            for voc in self.indicator_vocs:
                features[f'voc_{voc}'] = 1 if voc in sample_vocs else 0
            
            feature_data.append(features)
        
        X_voc = pd.DataFrame(feature_data)
        X = X_voc[[col for col in X_voc.columns if col.startswith('voc_')]]
        y = df_samples['class_label'].values
        
        print(f"[OK] Created feature matrix: {X.shape}")
        
        return X, y, df_samples, X_voc
    
    def evaluate_on_replicate(self, rep_name='R2'):
        """Evaluate R1-trained model on R2 data"""
        
        print(f"\n{'='*80}")
        print(f"EVALUATING R1 MODEL ON {rep_name} DATA")
        print(f"{'='*80}")
        
        # Load R2 data
        X, y, df_samples, X_voc = self.load_replicate_data(rep_name)
        
        # Scale using R1's scaler
        X_scaled = X.copy()
        numeric_cols = X.columns.tolist()
        X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Get predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        cm = confusion_matrix(y_encoded, y_pred)
        accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average=None, zero_division=0)
        recall = recall_score(y_encoded, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_encoded, y_pred, average=None, zero_division=0)
        
        print(f"\n{'='*80}")
        print(f"{rep_name} EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Recall (Spoilage): {recall[-1]:.4f}")
        print(f"Precision (Avg): {precision.mean():.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_encoded, y_pred,
                                   target_names=self.label_encoder.classes_,
                                   digits=3))
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store results
        results = {
            'replicate': rep_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_true': y,
            'y_pred': self.label_encoder.inverse_transform(y_pred),
            'y_proba': y_proba,
            'samples': df_samples,
            'classes': self.label_encoder.classes_
        }
        
        print(f"\n[OK] Evaluation complete for {rep_name}")
        
        return results


def run_extended_validation():
    """Run extended validation on R2 and R3"""
    
    print("\n" + "="*80)
    print("EXTENDED VALIDATION: R1 MODEL ON R2 AND R3 DATA")
    print("="*80)
    
    validator = ExtendedValidator()
    
    # Evaluate on R2
    r2_results = validator.evaluate_on_replicate('R2')
    
    # Evaluate on R3
    print("\n" + "-"*80)
    r3_results = validator.evaluate_on_replicate('R3')
    
    all_results = {
        'R2': r2_results,
        'R3': r3_results
    }
    
    print("\n" + "="*80)
    print("EXTENDED VALIDATION COMPLETE")
    print("="*80)
    
    return all_results
