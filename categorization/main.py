"""
VOC-Based Chicken Spoilage Categorization
==========================================
Simple pipeline: Load data -> Create VOC features -> Train RF -> Show results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from categorization import DataProcessor, CategorizationModel
from visualization import VisualizationEngine


def main():
    """Execute VOC-based categorization pipeline"""
    
    print("="*70)
    print("VOC-BASED CHICKEN SPOILAGE CATEGORIZATION")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD AND PROCESS DATA
    # =========================================================================
    print("\n[STEP 1] Loading raw VOC data...")
    print("-" * 70)
    
    processor = DataProcessor(data_path="../data/raw_data/DataAI.csv")
    processor.load_data()
    processor.clean_data()
    X, y = processor.get_features()
    
    print(f"Classes: {y.unique()}")
    print(f"Distribution:\n{y.value_counts()}")
    
    # =========================================================================
    # STEP 2: TRAIN MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 2] Training Random Forest on VOC features...")
    print("-" * 70)
    
    model = CategorizationModel(random_state=42)
    model.prepare_data(X, y)
    model.train()
    
    # =========================================================================
    # STEP 3: EVALUATE
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 3] Evaluating model...")
    print("-" * 70)
    
    model.evaluate()
    
    # =========================================================================
    # STEP 4: OPTIMIZE THRESHOLD
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 4] Optimizing threshold for high spoilage recall...")
    print("-" * 70)
    
    model.optimize_threshold(target_recall=0.95)
    
    # =========================================================================
    # STEP 5: SHOW TOP PREDICTIVE VOCs
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 5] TOP VOCs FOR SPOILAGE PREDICTION")
    print("="*70)
    
    importance_df = model.get_feature_importance()
    print("\nTop 10 Most Important VOCs:")
    for idx, row in importance_df.head(10).iterrows():
        voc_name = row['voc'].replace('voc_', '')
        print(f"  {voc_name:40s}: {row['importance']:.4f}")
    
    # =========================================================================
    # STEP 6: VISUALIZE
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 6] Generating visualizations...")
    print("-" * 70)
    
    viz = VisualizationEngine(model)
    viz.plot_class_distribution()
    viz.plot_confusion_matrices()
    viz.plot_feature_importance()
    viz.plot_threshold_optimization()
    
    # =========================================================================
    # STEP 7: SAVE
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 7] Saving model...")
    print("-" * 70)
    
    model.save_model(output_dir='../model_pkls')
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("CATEGORIZATION COMPLETE")
    print("="*70)
    print(f"""
Model Performance (Test Set):
  - Accuracy: {model.evaluation_results['accuracy']:.1%}
  - Spoilage Recall: {model.evaluation_results['recall_spoiled']:.1%} (catches spoilage)
  - Optimal Threshold: {model.optimal_threshold:.2f}

Key Findings:
  - Top spoilage VOCs: {', '.join(importance_df.head(3)['voc'].str.replace('voc_', ''))}
  - Model is regularized to prevent overfitting
  - 70-15-15 train-val-test split used

Output:
  - Visualizations: categorization/figure/
  - Model: model_pkls/categorization_model.pkl
""")
    
    return model


if __name__ == "__main__":
    model = main()

