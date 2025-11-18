"""
Recall Optimization Module
Threshold tuning to maximize recall for spoilage detection.
"""

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


class RecallOptimizer:
    """Optimize model threshold for food safety (high recall)."""
    
    def __init__(self, spoilage_class=2):
        """
        Initialize optimizer.
        
        Args:
            spoilage_class: Class index for spoilage (default 2)
        """
        self.spoilage_class = spoilage_class
        self.optimal_threshold = 0.5
        self.optimization_results = None
    
    def find_optimal_threshold(self, y_val, y_val_proba, target_recall=0.95, thresholds=None):
        """
        Find optimal threshold to achieve target recall.
        
        Args:
            y_val: True validation labels
            y_val_proba: Predicted probabilities from model
            target_recall: Target recall for spoilage class (default 0.95 = 95%)
            thresholds: Thresholds to test (default: 0.1-0.9)
            
        Returns:
            Optimal threshold and metrics dict
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.05)
        
        print("\n" + "=" * 80)
        print("RECALL OPTIMIZATION FOR SPOILAGE DETECTION")
        print("=" * 80)
        print(f"\nTarget Recall: {target_recall:.1%}")
        print(f"Target Class: Spoilage (index {self.spoilage_class})")
        
        results = []
        best_threshold = 0.5
        best_diff = float('inf')
        
        print(f"\nTesting thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
        print(f"\n{'Threshold':<12} {'Recall':<12} {'Precision':<12} {'F1':<12}")
        print("-" * 50)
        
        for threshold in thresholds:
            # Create binary predictions: spoilage vs not-spoilage
            y_val_pred_binary = (y_val_proba[:, self.spoilage_class] >= threshold).astype(int)
            
            # Convert back to multi-class: if not spoilage, predict most likely non-spoilage
            y_val_pred = y_val.copy()
            not_spoilage_mask = y_val_pred_binary == 0
            
            if not_spoilage_mask.sum() > 0:
                # For non-spoilage samples, predict class with highest prob (excluding spoilage)
                proba_without_spoilage = y_val_proba[:, :self.spoilage_class].copy()
                y_val_pred[not_spoilage_mask] = proba_without_spoilage[not_spoilage_mask].argmax(axis=1)
            
            # For spoilage samples, predict spoilage class
            y_val_pred[y_val_pred_binary == 1] = self.spoilage_class
            
            # Calculate metrics for spoilage class
            recall = recall_score(y_val, y_val_pred, labels=[self.spoilage_class], 
                                average='binary', zero_division=0)
            precision = precision_score(y_val, y_val_pred, labels=[self.spoilage_class],
                                      average='binary', zero_division=0)
            f1 = f1_score(y_val, y_val_pred, labels=[self.spoilage_class],
                         average='binary', zero_division=0)
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'predictions': y_val_pred
            })
            
            print(f"{threshold:<12.2f} {recall:<12.4f} {precision:<12.4f} {f1:<12.4f}")
            
            # Track threshold closest to target recall
            diff = abs(recall - target_recall)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        
        self.optimization_results = results
        self.optimal_threshold = best_threshold
        
        best_result = next((r for r in results if r['threshold'] == best_threshold), None)
        
        print("\n" + "=" * 80)
        print(f"OPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")
        print("=" * 80)
        if best_result:
            print(f"  Recall:    {best_result['recall']:.4f} ({best_result['recall']*100:.1f}%)")
            print(f"  Precision: {best_result['precision']:.4f}")
            print(f"  F1-Score:  {best_result['f1']:.4f}")
        
        return best_threshold, best_result
    
    def apply_threshold(self, y_proba, threshold=None):
        """
        Apply optimal threshold to predictions.
        
        Args:
            y_proba: Predicted probabilities
            threshold: Threshold to apply (uses optimal if None)
            
        Returns:
            Binary predictions (0: not spoilage, 1: spoilage)
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        return (y_proba[:, self.spoilage_class] >= threshold).astype(int)
    
    def evaluate_threshold(self, y_true, y_true_binary, y_proba, threshold=None):
        """
        Evaluate model with threshold applied.
        
        Args:
            y_true: True multi-class labels
            y_true_binary: True binary labels (0: not spoilage, 1: spoilage)
            y_proba: Predicted probabilities
            threshold: Threshold to apply
            
        Returns:
            Metrics dict
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        y_pred_binary = self.apply_threshold(y_proba, threshold)
        
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        metrics = {
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'confusion_matrix': cm,
            'tp': cm[1, 1],
            'fp': cm[0, 1],
            'tn': cm[0, 0],
            'fn': cm[1, 0]
        }
        
        return metrics
    
    def print_threshold_analysis(self, metrics, set_name="Test"):
        """Print analysis of threshold application."""
        print("\n" + "=" * 80)
        print(f"THRESHOLD APPLICATION - {set_name.upper()} SET")
        print("=" * 80)
        print(f"\nThreshold: {metrics['threshold']:.2f}")
        print(f"\nBinary Metrics (Spoilage vs Not-Spoilage):")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['tn']} (correctly identified fresh/moderate)")
        print(f"  False Positives: {metrics['fp']} (incorrectly marked as spoiled)")
        print(f"  True Positives:  {metrics['tp']} (correctly identified spoiled)")
        print(f"  False Negatives: {metrics['fn']} (missed spoiled - FOOD SAFETY RISK!)")
        
        if metrics['fn'] > 0:
            print(f"\n⚠️  WARNING: {metrics['fn']} spoiled samples NOT detected!")
        else:
            print(f"\n✓ ALL spoiled samples detected (100% recall)")
    
    def get_optimization_results(self):
        """Get all optimization results."""
        return self.optimization_results
    
    def get_optimal_threshold(self):
        """Get optimal threshold."""
        return self.optimal_threshold
