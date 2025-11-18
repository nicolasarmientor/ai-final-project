"""
Main Orchestration Script
==========================
Executes the chicken spoilage categorization pipeline.

Pipeline Steps:
  1. Load raw data from DataAI.csv
  2. Process and aggregate VOC data per sample
  3. Train three classification models
  4. Evaluate models and select best performer
  5. Optimize decision threshold for 95%+ spoilage recall
  6. Visualize results
  7. Save production model

Commented placeholders for other models:
  - Multinomial Naive Bayes (alternative, simpler approach)
  - SVM with RBF kernel (non-linear alternative)
  - Neural Network (deep learning alternative)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from categorization.categorization import DataProcessor, CategorizationModels
from categorization.visualization import VisualizationEngine


def main():
    """Execute full categorization pipeline"""
    
    print("="*70)
    print("CHICKEN SPOILAGE CATEGORIZATION PIPELINE")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD AND PROCESS DATA
    # =========================================================================
    print("\n[STEP 1] Loading and processing raw VOC data...")
    print("-" * 70)
    
    processor = DataProcessor(data_path="data/raw_data/DataAI.csv")
    processor.load_data()
    processor.clean_data()
    X, y = processor.get_features()
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Classes: {y.unique()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # =========================================================================
    # STEP 2: TRAIN MODELS
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 2] Training classification models...")
    print("-" * 70)
    
    models = CategorizationModels(random_state=42)
    models.prepare_data(X, y)
    models.train_models()
    models.get_predictions()
    
    # =========================================================================
    # STEP 3: EVALUATE MODELS
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 3] Evaluating models...")
    print("-" * 70)
    
    models.evaluate()
    
    # =========================================================================
    # STEP 4: OPTIMIZE THRESHOLD FOR HIGH RECALL
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 4] Optimizing decision threshold for food safety...")
    print("-" * 70)
    
    models.optimize_threshold(target_recall=0.95)
    
    # =========================================================================
    # STEP 5: VISUALIZE RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 5] Generating visualizations...")
    print("-" * 70)
    
    viz = VisualizationEngine(models)
    viz.plot_class_distribution()
    viz.plot_confusion_matrices()
    viz.plot_model_comparison()
    viz.plot_feature_importance()
    viz.plot_threshold_optimization()
    
    # =========================================================================
    # STEP 6: SAVE PRODUCTION MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 6] Saving production model...")
    print("-" * 70)
    
    models.save_best_model(output_dir='model_pkls')
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"""
Results Summary:
  - Best Model: Random Forest (with threshold optimization)
  - Optimal Threshold: {models.optimal_threshold:.2f}
  - Spoilage Recall: {models.evaluation_results['RandomForest (Optimized)']['recall_spoiled']:.4f}
  - Spoilage Precision: {models.evaluation_results['RandomForest (Optimized)']['precision_spoiled']:.4f}
  
Output:
  - Visualizations saved to: categorization/figure/
  - Model saved to: model_pkls/categorization_model.pkl
  
Food Safety Note:
  The model prioritizes recall (catching spoiled samples) over precision.
  Confidence scores < 0.30 should trigger manual inspection.
""")
    
    return models


# ============================================================================
# ALTERNATIVE MODEL IMPLEMENTATIONS (Commented - For Reference)
# ============================================================================

"""
MULTINOMIAL NAIVE BAYES (Simpler Alternative)
================================================
from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, y_train):
    '''Simple, fast model - good baseline for comparison'''
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb

# Usage: nb_model = train_naive_bayes(X_train_scaled, y_train_encoded)


SUPPORT VECTOR MACHINE - RBF Kernel (Non-Linear Alternative)
==============================================================
from sklearn.svm import SVC

def train_svm_rbf(X_train, y_train):
    '''Non-linear SVM - captures complex patterns but slower'''
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
              probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm

# Usage: svm_model = train_svm_rbf(X_train_scaled, y_train_encoded)


NEURAL NETWORK (Deep Learning Alternative)
============================================
from sklearn.neural_network import MLPClassifier

def train_neural_network(X_train, y_train):
    '''Neural network - captures non-linear patterns at computational cost'''
    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                       early_stopping=True, validation_fraction=0.2,
                       random_state=42)
    nn.fit(X_train, y_train)
    return nn

# Usage: nn_model = train_neural_network(X_train_scaled, y_train_encoded)
"""


if __name__ == "__main__":
    models = main()
