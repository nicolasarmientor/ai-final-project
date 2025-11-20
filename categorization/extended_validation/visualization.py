"""
Extended Validation Visualization
==================================
Generate same figures as regular categorization for extended validation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_class_distribution(results_dict, output_dir='figures'):
    """Plot class distribution for each replicate"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5*len(results_dict), 4))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for ax, (rep_name, results) in zip(axes, results_dict.items()):
        classes = results['classes']
        y_true = results['y_true']
        
        class_counts = pd.Series(y_true).value_counts().sort_index()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        ax.bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)])
        ax.set_ylabel('Count')
        ax.set_title(f'{rep_name} Class Distribution')
        ax.set_ylim([0, max(class_counts.values) * 1.1])
        
        for i, v in enumerate(class_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    plt.suptitle('Class Distribution - Extended Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: class_distribution.png")
    plt.close()


def plot_confusion_matrices(results_dict, output_dir='figures'):
    """Plot confusion matrices for each replicate"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 4))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for ax, (rep_name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        classes = results['classes']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=classes, yticklabels=classes)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        ax.set_title(f'{rep_name} Confusion Matrix')
    
    plt.suptitle('Confusion Matrices - Extended Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: confusion_matrix.png")
    plt.close()


def plot_metrics_comparison(results_dict, output_dir='figures'):
    """Plot accuracy, precision, recall, F1 across replicates"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    for rep_name, results in results_dict.items():
        accuracy = results['accuracy']
        precision_mean = np.mean(results['precision'])
        recall_mean = np.mean(results['recall'])
        f1_mean = np.mean(results['f1'])
        
        metrics_data.append({
            'Replicate': rep_name,
            'Accuracy': accuracy,
            'Precision': precision_mean,
            'Recall': recall_mean,
            'F1-Score': f1_mean
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, metric in zip(axes.flat, metrics):
        ax.bar(metrics_df['Replicate'], metrics_df[metric], color=colors[:len(metrics_df)])
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Replicate')
        ax.set_ylim([0, 1.1])
        
        for i, v in enumerate(metrics_df[metric]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle('Performance Metrics - Extended Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: metrics_comparison.png")
    plt.close()
    
    return metrics_df


def plot_recall_by_class(results_dict, output_dir='figures'):
    """Plot recall for each class"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 4))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for ax, (rep_name, results) in zip(axes, results_dict.items()):
        classes = results['classes']
        recall = results['recall']
        
        ax.bar(classes, recall, color=colors[:len(classes)])
        ax.set_ylabel('Recall')
        ax.set_title(f'{rep_name} Recall by Class')
        ax.set_ylim([0, 1.1])
        
        for i, v in enumerate(recall):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle('Recall by Class - Extended Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'recall_by_class.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: recall_by_class.png")
    plt.close()


def generate_all_visualizations(results_dict, output_dir='figures'):
    """Generate all visualization plots"""
    
    print(f"\n{'='*80}")
    print(f"GENERATING VISUALIZATIONS FOR EXTENDED VALIDATION")
    print(f"{'='*80}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plot_class_distribution(results_dict, output_dir)
    plot_confusion_matrices(results_dict, output_dir)
    metrics_df = plot_metrics_comparison(results_dict, output_dir)
    plot_recall_by_class(results_dict, output_dir)
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"[OK] All figures saved to: {output_dir}/")
    
    return metrics_df
