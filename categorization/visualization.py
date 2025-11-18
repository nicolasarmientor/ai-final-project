"""
Visualization Engine
====================
Generates publication-quality plots for model results.

Output:
  - figure/01_class_distribution.png
  - figure/02_confusion_matrices.png
  - figure/03_model_comparison.png
  - figure/04_feature_importance.png
  - figure/05_threshold_optimization.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class VisualizationEngine:
    """Generate and save visualization plots"""
    
    def __init__(self, models, figure_dir='categorization/figure'):
        self.models = models
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_class_distribution(self):
        """Plot class distribution across train/val/test splits"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        splits = {
            'Training': self.models.train_data['y_orig'],
            'Validation': self.models.val_data['y_orig'],
            'Test': self.models.test_data['y_orig']
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
        
        print(f"✓ Saved: 01_class_distribution.png")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        model_names = ['LogisticRegression', 'RandomForest', 'XGBoost']
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        
        class_labels = self.models.label_encoder.classes_
        
        for idx, model_name in enumerate(model_names):
            cm = self.models.evaluation_results[model_name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=class_labels, yticklabels=class_labels,
                       ax=axes[idx], cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{model_name}', fontweight='bold', fontsize=11)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.suptitle('Confusion Matrices (Test Set)', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figure_dir / '02_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 02_confusion_matrices.png")
    
    def plot_model_comparison(self):
        """Compare models on key metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = ['LogisticRegression', 'RandomForest', 'XGBoost']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        results = {}
        for model_name in model_names:
            eval_res = self.models.evaluation_results[model_name]
            results[model_name] = {
                'accuracy': eval_res['accuracy'],
                'precision': eval_res['precision'].mean(),
                'recall': eval_res['recall'].mean(),
                'f1': eval_res['f1'].mean()
            }
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[m][metric] for m in model_names]
            
            bars = ax.bar(model_names, values, color=colors, edgecolor='black')
            ax.set_ylabel(metric.capitalize(), fontweight='bold')
            ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold', fontsize=11)
            ax.set_ylim([0, 1.05])
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Focus on Spoilage Recall
        ax = axes[1, 1]
        spoilage_recalls = [self.models.evaluation_results[m]['recall_spoiled'] for m in model_names]
        bars = ax.bar(model_names, spoilage_recalls, color=colors, edgecolor='black')
        ax.set_ylabel('Recall Score', fontweight='bold')
        ax.set_title('Spoilage Recall (Critical)', fontweight='bold', fontsize=11, color='red')
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Target (0.95)')
        ax.legend()
        
        for bar, val in zip(bars, spoilage_recalls):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.figure_dir / '03_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 03_model_comparison.png")
    
    def plot_feature_importance(self):
        """Plot feature importance from best models"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Random Forest
        rf_model = self.models.models['RandomForest']
        rf_importance = rf_model.feature_importances_
        feature_names = self.models.feature_names
        
        rf_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_importance
        }).sort_values('Importance', ascending=True)
        
        axes[0].barh(rf_df['Feature'], rf_df['Importance'], color='#2ecc71', edgecolor='black')
        axes[0].set_title('Random Forest - Feature Importance', fontweight='bold', fontsize=11)
        axes[0].set_xlabel('Importance Score', fontweight='bold')
        
        # XGBoost
        xgb_model = self.models.models['XGBoost']
        xgb_importance = xgb_model.feature_importances_
        
        xgb_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_importance
        }).sort_values('Importance', ascending=True)
        
        axes[1].barh(xgb_df['Feature'], xgb_df['Importance'], color='#e74c3c', edgecolor='black')
        axes[1].set_title('XGBoost - Feature Importance', fontweight='bold', fontsize=11)
        axes[1].set_xlabel('Importance Score', fontweight='bold')
        
        plt.suptitle('Feature Importance Across Models', fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.figure_dir / '04_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 04_feature_importance.png")
    
    def plot_threshold_optimization(self):
        """Plot threshold optimization for high spoilage recall"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract probabilities and labels
        rf_test_proba = self.models.predictions['RandomForest']['test_proba']
        y_test = self.models.test_data['y']
        
        spoiled_class_idx = len(self.models.label_encoder.classes_) - 1
        rf_test_proba_spoiled = rf_test_proba[:, spoiled_class_idx]
        
        spoiled_mask = y_test == spoiled_class_idx
        non_spoiled_mask = y_test != spoiled_class_idx
        
        # Plot probability distributions
        ax.hist(rf_test_proba_spoiled[spoiled_mask], bins=20, alpha=0.6, 
               label='Actual Spoiled', color='#e74c3c', edgecolor='black')
        ax.hist(rf_test_proba_spoiled[non_spoiled_mask], bins=20, alpha=0.6,
               label='Actual Fresh/Moderate', color='#2ecc71', edgecolor='black')
        
        # Mark optimal threshold
        ax.axvline(x=self.models.optimal_threshold, color='#f39c12', linestyle='--',
                  linewidth=3, label=f'Optimal Threshold ({self.models.optimal_threshold:.2f})')
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7,
                  label='Default Threshold (0.5)')
        
        ax.set_xlabel('Predicted Probability of Spoilage', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Threshold Optimization: Probability Distributions by True Class\n' +
                    '(Lower threshold → Higher recall, Prevent food poisoning)', 
                    fontweight='bold', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / '05_threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 05_threshold_optimization.png")


# Import pandas for feature importance visualization
import pandas as pd
