"""
Visualization Engine - VOC-Based Spoilage Detection
=====================================================
Generates plots for Random Forest classification results.

Output files:
  - figure/01_class_distribution.png
  - figure/02_confusion_matrices.png
  - figure/03_feature_importance.png
  - figure/04_threshold_optimization.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


class VisualizationEngine:
    """Generate and save visualization plots"""
    
    def __init__(self, model, figure_dir='figure'):
        self.model = model
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(exist_ok=True, parents=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_class_distribution(self):
        """Plot class distribution across train/val/test splits"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        splits = {
            'Training': self.model.train_data['y_orig'],
            'Validation': self.model.val_data['y_orig'],
            'Test': self.model.test_data['y_orig']
        }
        
        colors = {'fresh': '#2ecc71', 'moderate': '#f39c12', 'spoiled': '#e74c3c'}
        
        for idx, (split_name, y_split) in enumerate(splits.items()):
            counts = y_split.value_counts()
            color_list = [colors.get(label, '#3498db') for label in counts.index]
            
            axes[idx].bar(counts.index, counts.values, color=color_list, edgecolor='black')
            axes[idx].set_title(f'{split_name} Set\n(n={len(y_split)})', fontweight='bold', fontsize=11)
            axes[idx].set_ylabel('Count')
            axes[idx].set_ylim([0, max(counts.values) * 1.15])
            
            for i, v in enumerate(counts.values):
                pct = (v / len(y_split)) * 100
                axes[idx].text(i, v + 1, f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=9)
        
        plt.suptitle('Class Distribution Across Splits', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figure_dir / '01_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Saved: 01_class_distribution.png")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrix for Random Forest"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = self.model.evaluation_results['confusion_matrix']
        class_labels = self.model.label_encoder.classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=class_labels, yticklabels=class_labels,
                   ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 12})
        
        ax.set_title('Random Forest - Confusion Matrix (Test Set)', fontweight='bold', fontsize=13)
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / '02_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Saved: 02_confusion_matrices.png")
    
    def plot_feature_importance(self):
        """Plot top VOC features by importance"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        importance_df = self.model.get_feature_importance().head(15)
        
        # Clean up VOC names
        importance_df['voc_clean'] = importance_df['voc'].str.replace('voc_', '')
        
        # Sort and plot
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        colors = ['#e74c3c' if imp > 0.10 else '#f39c12' if imp > 0.05 else '#3498db' 
                 for imp in importance_df['importance']]
        
        ax.barh(importance_df['voc_clean'], importance_df['importance'], 
               color=colors, edgecolor='black')
        ax.set_xlabel('Importance Score', fontweight='bold', fontsize=11)
        ax.set_title('Top 15 VOCs - Feature Importance for Spoilage Detection', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            ax.text(row['importance'] + 0.002, i, f"{row['importance']:.4f}", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / '03_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Saved: 03_feature_importance.png")
    
    def plot_threshold_optimization(self):
        """Plot threshold optimization for spoilage recall"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract probabilities and labels
        rf_test_proba = self.model.predictions['test_proba']
        y_test = self.model.test_data['y']
        
        # Get spoiled class index
        spoiled_class_idx = len(self.model.label_encoder.classes_) - 1
        rf_test_proba_spoiled = rf_test_proba[:, spoiled_class_idx]
        
        spoiled_mask = y_test == spoiled_class_idx
        non_spoiled_mask = y_test != spoiled_class_idx
        
        # PLOT 1: Probability distributions
        ax1.hist(rf_test_proba_spoiled[spoiled_mask], bins=15, alpha=0.6, 
                label='Actual Spoiled', color='#e74c3c', edgecolor='black', linewidth=1.5)
        ax1.hist(rf_test_proba_spoiled[non_spoiled_mask], bins=15, alpha=0.6,
                label='Actual Fresh/Moderate', color='#2ecc71', edgecolor='black', linewidth=1.5)
        
        # Mark optimal threshold
        ax1.axvline(x=self.model.optimal_threshold, color='#f39c12', linestyle='--',
                   linewidth=3, label=f'Optimal: {self.model.optimal_threshold:.2f}')
        ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7,
                   label='Default: 0.5')
        
        ax1.set_xlabel('Predicted Probability of Spoilage', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Probability Distribution by True Class', fontweight='bold', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # PLOT 2: Metrics vs threshold
        thresholds = np.linspace(0, 1, 50)
        recalls = []
        precisions = []
        
        for thresh in thresholds:
            y_pred_thresh = (rf_test_proba_spoiled >= thresh).astype(int)
            
            tp = np.sum((y_pred_thresh == 1) & (y_test == spoiled_class_idx))
            fn = np.sum((y_pred_thresh == 0) & (y_test == spoiled_class_idx))
            fp = np.sum((y_pred_thresh == 1) & (y_test != spoiled_class_idx))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            recalls.append(recall)
            precisions.append(precision)
        
        ax2.plot(thresholds, recalls, marker='o', linewidth=2, markersize=4,
                label='Recall (Sensitivity)', color='#e74c3c')
        ax2.plot(thresholds, precisions, marker='s', linewidth=2, markersize=4,
                label='Precision', color='#2ecc71')
        
        ax2.axvline(x=self.model.optimal_threshold, color='#f39c12', linestyle='--',
                   linewidth=2.5, label=f'Optimal Threshold')
        ax2.axhline(y=0.95, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax2.set_xlabel('Decision Threshold', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Metrics vs. Threshold', fontweight='bold', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.suptitle('Threshold Optimization for High Spoilage Recall', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.figure_dir / '04_threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Saved: 04_threshold_optimization.png")
