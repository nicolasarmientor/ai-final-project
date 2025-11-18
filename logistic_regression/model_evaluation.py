"""
Model Evaluation Module
Compute metrics and generate evaluation reports.
"""

from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, 
    recall_score, f1_score, accuracy_score
)
import numpy as np


class ModelEvaluator:
    """Evaluate model performance with detailed metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def evaluate_model(self, model_name, y_true, y_pred, set_name="Test"):
        """
        Evaluate model on a single set.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            set_name: Name of the set (Train/Val/Test)
            
        Returns:
            Dict with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, zero_division=0)
        }
        
        if model_name not in self.results:
            self.results[model_name] = {}
        
        self.results[model_name][set_name] = metrics
        
        return metrics
    
    def evaluate_all_sets(self, model_name, y_train, y_val, y_test, 
                          y_train_pred, y_val_pred, y_test_pred):
        """Evaluate model on all sets."""
        print(f"\nEvaluating {model_name}...")
        
        self.evaluate_model(model_name, y_train, y_train_pred, "Train")
        self.evaluate_model(model_name, y_val, y_val_pred, "Val")
        self.evaluate_model(model_name, y_test, y_test_pred, "Test")
        
        return self.results[model_name]
    
    def print_summary(self, model_name):
        """Print evaluation summary."""
        if model_name not in self.results:
            print(f"No results for {model_name}")
            return
        
        print("\n" + "=" * 80)
        print(f"EVALUATION SUMMARY: {model_name.upper()}")
        print("=" * 80)
        
        for set_name, metrics in self.results[model_name].items():
            print(f"\n{set_name} Set:")
            print(f"  Accuracy:        {metrics['accuracy']:.4f}")
            print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
            print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
            print(f"\n  Classification Report:")
            for line in metrics['classification_report'].split('\n'):
                if line.strip():
                    print(f"    {line}")
    
    def compare_models(self, models_list=None):
        """Compare performance across models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON - TEST SET PERFORMANCE")
        print("=" * 80)
        
        if models_list is None:
            models_list = list(self.results.keys())
        
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 60)
        
        comparison_data = {}
        for model_name in models_list:
            if model_name not in self.results:
                continue
            
            if 'Test' not in self.results[model_name]:
                continue
            
            metrics = self.results[model_name]['Test']
            comparison_data[model_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision_macro'],
                'recall': metrics['recall_macro'],
                'f1': metrics['f1_macro']
            }
            
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} "
                  f"{metrics['precision_macro']:<12.4f} {metrics['recall_macro']:<12.4f} "
                  f"{metrics['f1_macro']:<12.4f}")
        
        # Find best model
        if comparison_data:
            best_model = max(comparison_data.items(), 
                           key=lambda x: x[1]['f1'])
            print(f"\n✓ Best Model (by F1): {best_model[0]}")
        
        return comparison_data
    
    def get_per_class_metrics(self, model_name, y_true, y_pred, class_names=None):
        """Get per-class metrics."""
        if class_names is None:
            class_names = {0: 'Fresh', 1: 'Moderate', 2: 'Spoiled'}
        
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{model_name} - Per-Class Metrics:")
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 50)
        
        for class_idx in range(len(class_names)):
            precision = precision_score(y_true, y_pred, labels=[class_idx], 
                                      average='micro', zero_division=0)
            recall = recall_score(y_true, y_pred, labels=[class_idx], 
                                average='micro', zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=[class_idx], 
                         average='micro', zero_division=0)
            
            print(f"{class_names.get(class_idx, f'Class {class_idx}'):<12} "
                  f"{precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        return cm
    
    def print_confusion_matrix(self, model_name, set_name="Test"):
        """Print confusion matrix."""
        if model_name not in self.results or set_name not in self.results[model_name]:
            return
        
        cm = self.results[model_name][set_name]['confusion_matrix']
        
        print(f"\n{model_name} - Confusion Matrix ({set_name}):")
        print("      Predicted:")
        print("       Fresh  Moderate  Spoiled")
        
        class_names = ['Fresh', 'Moderate', 'Spoiled']
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<6} {cm[i][0]:<6} {cm[i][1]:<9} {cm[i][2]:<8}")
    
    def get_results(self):
        """Get all results."""
        return self.results
