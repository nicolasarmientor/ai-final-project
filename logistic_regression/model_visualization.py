"""
Visualization Module
Generate analysis plots and charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


class ModelVisualizer:
    """Generate visualizations for analysis."""
    
    def __init__(self, output_dir="logistic_regression/figure"):
        """Initialize visualizer."""
        self.output_dir = output_dir
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_class_distribution(self, y_data, title="Class Distribution"):
        """Plot class distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_counts = y_data.value_counts().sort_index()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        class_names = ['Fresh', 'Moderate', 'Spoiled']
        
        bars = ax.bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels([class_names[i] for i in class_counts.index])
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", 
                            normalize=True):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        class_names = ['Fresh', 'Moderate', 'Spoiled']
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=True, ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, comparison_data):
        """Plot model comparison."""
        models = list(comparison_data.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_names):
            values = [comparison_data[m][metric] for m in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=10):
        """Plot feature importance from Random Forest or XGBoost."""
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {type(model).__name__} does not support feature importance")
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(indices)), importances[indices], alpha=0.8, color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_tradeoff(self, optimization_results):
        """Plot precision-recall tradeoff across thresholds."""
        thresholds = [r['threshold'] for r in optimization_results]
        recalls = [r['recall'] for r in optimization_results]
        precisions = [r['precision'] for r in optimization_results]
        f1s = [r['f1'] for r in optimization_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precision-Recall tradeoff
        ax1.plot(thresholds, recalls, 'o-', label='Recall', linewidth=2, markersize=6)
        ax1.plot(thresholds, precisions, 's-', label='Precision', linewidth=2, markersize=6)
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target Recall (95%)')
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1-Score by threshold
        ax2.plot(thresholds, f1s, 'D-', color='green', linewidth=2, markersize=6)
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('F1-Score by Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_style_threshold_analysis(self, optimization_results):
        """Plot recall vs false positive rate across thresholds."""
        recalls = [r['recall'] for r in optimization_results]
        precisions = [r['precision'] for r in optimization_results]
        thresholds = [r['threshold'] for r in optimization_results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(1 - np.array(precisions), recalls, c=thresholds, 
                           cmap='viridis', s=100, alpha=0.7)
        
        for i, threshold in enumerate(thresholds):
            ax.annotate(f'{threshold:.2f}', 
                       (1 - precisions[i], recalls[i]),
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('False Positive Rate (1 - Precision)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('Threshold Analysis: Recall vs FPR', fontsize=14, fontweight='bold')
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Threshold', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_all_confusion_matrices(self, models_dict, y_val, y_test):
        """Plot confusion matrices for all models."""
        num_models = len(models_dict)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (model_name, predictions) in enumerate(models_dict.items()):
            val_pred = predictions['val']
            test_pred = predictions['test']
            
            cm_test = confusion_matrix(y_test, test_pred)
            cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
            
            class_names = ['Fresh', 'Moderate', 'Spoiled']
            sns.heatmap(cm_test_norm, annot=True, fmt='.2f', cmap='Blues', 
                       cbar=True, ax=axes[idx],
                       xticklabels=class_names, yticklabels=class_names)
            
            axes[idx].set_title(f'{model_name} (Test Set)', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide last subplot if odd number of models
        if num_models < 4:
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename):
        """Save figure to file."""
        filepath = f"{self.output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close(fig)
