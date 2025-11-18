"""
Data Processor Module
Handles data loading, cleaning, filtering, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataProcessor:
    """Process and prepare data for modeling."""
    
    def __init__(self, data_path="data/processed_data/logistic_regression_data.csv"):
        """
        Initialize data processor.
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
    
    def load_data(self):
        """Load data from CSV file."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"\n✓ Data loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            raise
    
    def display_overview(self):
        """Display data overview."""
        print("\n" + "=" * 80)
        print("DATA OVERVIEW")
        print("=" * 80)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        
        print(f"\nClass Distribution:")
        print(self.df['class_label'].value_counts())
    
    def clean_data(self):
        """Clean and preprocess data."""
        print("\n" + "=" * 80)
        print("DATA CLEANING")
        print("=" * 80)
        
        self.df_processed = self.df.copy()
        
        # Remove missing values
        initial_rows = len(self.df_processed)
        self.df_processed = self.df_processed.dropna()
        removed_nan = initial_rows - len(self.df_processed)
        print(f"\n1. Removed NaN values: {removed_nan} rows")
        
        # Filter by VOC Relevance Index (>= 80)
        initial_rows = len(self.df_processed)
        self.df_processed = self.df_processed[self.df_processed['revalence_index'] >= 80]
        removed_low_relevance = initial_rows - len(self.df_processed)
        
        print(f"\n2. VOC Relevance Index Filtering (>= 80):")
        print(f"   Removed: {removed_low_relevance} rows")
        print(f"   Retained: {len(self.df_processed)} rows")
        print(f"   Retention rate: {(len(self.df_processed)/len(self.df)*100):.2f}%")
        
        # Clean treatment names (remove spaces)
        self.df_processed['treatment'] = self.df_processed['treatment'].str.strip()
        print(f"\n3. Treatment names cleaned")
        print(f"   Unique treatments: {self.df_processed['treatment'].unique().tolist()}")
        
        # Verify class distribution
        print(f"\n4. Final Class Distribution:")
        print(self.df_processed['class_label'].value_counts())
        print("\n   Proportions (%):")
        print((self.df_processed['class_label'].value_counts(normalize=True) * 100).round(2))
        
        print(f"\n✓ Data cleaning complete!")
        print(f"  Final shape: {self.df_processed.shape}")
        
        return self.df_processed
    
    def get_processed_data(self):
        """Return processed data."""
        if self.df_processed is None:
            raise ValueError("Data not processed yet. Call clean_data() first.")
        return self.df_processed
    
    def save_processed_data(self, output_path="data/processed_data/categorization_data.csv"):
        """Save processed data to CSV."""
        if self.df_processed is None:
            raise ValueError("Data not processed yet. Call clean_data() first.")
        
        self.df_processed.to_csv(output_path, index=False)
        print(f"\n✓ Processed data saved to: {output_path}")
