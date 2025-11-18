"""
Feature Engineering Module
Handles feature creation, aggregation, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


class FeatureEngineer:
    """Create and engineer features for modeling."""
    
    def __init__(self, df):
        """
        Initialize feature engineer.
        
        Args:
            df (pd.DataFrame): Input dataframe with raw features
        """
        self.df = df.copy()
        self.df_features = None
        self.numeric_features = ['day', 'revalence_index', 'voc_count', 'voc_diversity_ratio']
        self.categorical_features = ['treatment']
    
    def engineer_features(self):
        """Create engineered features."""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        self.df_features = self.df.copy()
        
        # 1. Create VOC diversity ratio (normalize VOC count to 0-1)
        voc_min = self.df_features['voc_count'].min()
        voc_max = self.df_features['voc_count'].max()
        self.df_features['voc_diversity_ratio'] = (
            (self.df_features['voc_count'] - voc_min) / (voc_max - voc_min)
        )
        
        print(f"\n1. VOC Diversity Ratio Created:")
        print(f"   VOC count range: [{voc_min}, {voc_max}]")
        print(f"   Ratio range: [0, 1] (normalized)")
        print(f"   Sample values:")
        print(self.df_features[['voc_count', 'voc_diversity_ratio']].head())
        
        # 2. Display feature statistics
        print(f"\n2. Feature Statistics:")
        print(self.df_features[self.numeric_features].describe())
        
        # 3. Display categorical features
        print(f"\n3. Categorical Features:")
        for cat_feat in self.categorical_features:
            print(f"   {cat_feat}: {self.df_features[cat_feat].unique().tolist()}")
        
        print(f"\n✓ Feature engineering complete!")
        return self.df_features
    
    def get_features(self):
        """Return engineered features dataframe."""
        if self.df_features is None:
            raise ValueError("Features not engineered yet. Call engineer_features() first.")
        return self.df_features
    
    def get_X_y(self):
        """Separate features and target."""
        if self.df_features is None:
            raise ValueError("Features not engineered yet. Call engineer_features() first.")
        
        feature_cols = self.numeric_features + self.categorical_features
        X = self.df_features[feature_cols]
        y = self.df_features['class_label']
        
        print(f"\nFeature Matrix Shape: {X.shape}")
        print(f"Target Shape: {y.shape}")
        print(f"Features: {feature_cols}")
        
        return X, y


class PreprocessingPipeline:
    """Handle data preprocessing (encoding, scaling)."""
    
    def __init__(self, X_train, y_train):
        """
        Initialize preprocessing pipeline.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.X_train = X_train
        self.y_train = y_train
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
    
    def fit_preprocess(self):
        """Fit preprocessing on training data."""
        print("\n" + "=" * 80)
        print("PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # One-hot encode treatment
        print("\n1. One-hot encoding categorical features...")
        X_encoded = pd.get_dummies(self.X_train, columns=['treatment'], drop_first=True)
        self.feature_columns = X_encoded.columns
        
        # Scale numeric features
        print("2. Scaling numeric features...")
        numeric_cols = ['day', 'revalence_index', 'voc_count', 'voc_diversity_ratio']
        X_scaled = X_encoded.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X_encoded[numeric_cols])
        
        # Encode target
        print("3. Encoding target variable...")
        y_encoded = self.label_encoder.fit_transform(self.y_train)
        
        print(f"\n✓ Preprocessing pipeline fitted!")
        print(f"  Features shape: {X_scaled.shape}")
        print(f"  Class mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return X_scaled, y_encoded
    
    def transform(self, X, y=None):
        """Apply preprocessing to new data."""
        # One-hot encode
        X_encoded = pd.get_dummies(X, columns=['treatment'], drop_first=True)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.feature_columns]
        
        # Scale
        numeric_cols = ['day', 'revalence_index', 'voc_count', 'voc_diversity_ratio']
        X_scaled = X_encoded.copy()
        X_scaled[numeric_cols] = self.scaler.transform(X_encoded[numeric_cols])
        
        # Encode target if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
        
        return X_scaled, y_encoded
    
    def get_label_encoder(self):
        """Return fitted label encoder."""
        return self.label_encoder
    
    def get_scaler(self):
        """Return fitted scaler."""
        return self.scaler
