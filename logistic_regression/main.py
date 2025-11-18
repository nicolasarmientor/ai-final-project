"""
Main Orchestration Script
Runs complete pipeline: data loading → feature engineering → training → evaluation → optimization
"""

import sys
import os

# Import all modules
from data_processor import DataProcessor
from feature_engineering import FeatureEngineer, PreprocessingPipeline
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from recall_optimization import RecallOptimizer
from model_visualization import ModelVisualizer


def main():
    """Main pipeline execution."""
    
    print("\n" + "=" * 80)
    print(" " * 15 + "CHICKEN SPOILAGE CLASSIFICATION PIPELINE")
    print("=" * 80)
    
    # ==================== STEP 1: DATA LOADING ====================
    print("\n[STEP 1] Loading and Processing Data")
    print("-" * 80)
    
    data_processor = DataProcessor()
    df = data_processor.load_data()
    data_processor.display_overview()
    df_clean = data_processor.clean_data()
    
    # ==================== STEP 2: FEATURE ENGINEERING ====================
    print("\n[STEP 2] Feature Engineering")
    print("-" * 80)
    
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = feature_engineer.get_X_y(df_features)
    
    # ==================== STEP 3: TRAIN-VAL-TEST SPLIT ====================
    print("\n[STEP 3] Train-Validation-Test Split")
    print("-" * 80)
    
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # ==================== STEP 4: PREPROCESSING PIPELINE ====================
    print("\n[STEP 4] Preprocessing Pipeline (Encoding & Scaling)")
    print("-" * 80)
    
    preprocessor = PreprocessingPipeline()
    X_train_processed = preprocessor.fit_preprocess(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\nProcessed Data Shapes:")
    print(f"  Training:   {X_train_processed.shape}")
    print(f"  Validation: {X_val_processed.shape}")
    print(f"  Test:       {X_test_processed.shape}")
    
    # ==================== STEP 5: MODEL TRAINING ====================
    print("\n[STEP 5] Model Training")
    print("-" * 80)
    
    models = trainer.train_all_models(X_train_processed, y_train)
    
    # ==================== STEP 6: PREDICTIONS ====================
    print("\n[STEP 6] Generating Predictions")
    print("-" * 80)
    
    predictions = trainer.predict_all(
        X_train_processed, X_val_processed, X_test_processed,
        y_train, y_val, y_test
    )
    
    # ==================== STEP 7: MODEL EVALUATION ====================
    print("\n[STEP 7] Model Evaluation")
    print("-" * 80)
    
    evaluator = ModelEvaluator()
    
    for model_name, model_predictions in predictions.items():
        evaluator.evaluate_all_sets(
            model_name,
            y_train, y_val, y_test,
            model_predictions['train'], 
            model_predictions['val'],
            model_predictions['test']
        )
    
    # Print summaries
    for model_name in predictions.keys():
        evaluator.print_summary(model_name)
    
    # Compare models
    comparison_data = evaluator.compare_models()
    
    # ==================== STEP 8: RECALL OPTIMIZATION ====================
    print("\n[STEP 8] Recall Optimization for Food Safety")
    print("-" * 80)
    
    optimizer = RecallOptimizer(spoilage_class=2)
    
    # Use Random Forest for optimization (best model)
    rf_predictions = predictions.get('Random Forest')
    if rf_predictions:
        optimal_threshold, best_result = optimizer.find_optimal_threshold(
            y_val, 
            rf_predictions['val_proba'],
            target_recall=0.95
        )
        
        # Evaluate on test set with optimal threshold
        y_test_binary = (y_test == 2).astype(int)
        y_val_binary = (y_val == 2).astype(int)
        
        test_metrics = optimizer.evaluate_threshold(
            y_test, y_test_binary,
            rf_predictions['test_proba'],
            threshold=optimal_threshold
        )
        
        optimizer.print_threshold_analysis(test_metrics, "Test")
    
    # ==================== STEP 9: VISUALIZATIONS ====================
    print("\n[STEP 9] Generating Visualizations")
    print("-" * 80)
    
    visualizer = ModelVisualizer(output_dir="logistic_regression/figure")
    
    # Plot 1: Class distribution
    fig1 = visualizer.plot_class_distribution(y_train, "Class Distribution (Training Set)")
    visualizer.save_figure(fig1, "01_class_distribution_train.png")
    
    fig2 = visualizer.plot_class_distribution(y_test, "Class Distribution (Test Set)")
    visualizer.save_figure(fig2, "02_class_distribution_test.png")
    
    # Plot 2: Confusion matrices for all models
    fig3 = visualizer.plot_all_confusion_matrices(predictions, y_val, y_test)
    visualizer.save_figure(fig3, "03_confusion_matrices_all_models.png")
    
    # Plot 3: Model comparison
    fig4 = visualizer.plot_model_comparison(comparison_data)
    visualizer.save_figure(fig4, "04_model_comparison.png")
    
    # Plot 4: Feature importance (Random Forest)
    rf_model = models.get('Random Forest')
    if rf_model:
        feature_names = X_train_processed.columns if hasattr(X_train_processed, 'columns') else [f'Feature {i}' for i in range(X_train_processed.shape[1])]
        fig5 = visualizer.plot_feature_importance(rf_model, feature_names, top_n=10)
        if fig5:
            visualizer.save_figure(fig5, "05_feature_importance_rf.png")
    
    # Plot 5: Threshold optimization
    if optimizer.optimization_results:
        fig6 = visualizer.plot_precision_recall_tradeoff(optimizer.optimization_results)
        visualizer.save_figure(fig6, "06_precision_recall_tradeoff.png")
        
        fig7 = visualizer.plot_roc_style_threshold_analysis(optimizer.optimization_results)
        visualizer.save_figure(fig7, "07_threshold_analysis.png")
    
    # ==================== STEP 10: SAVE MODELS ====================
    print("\n[STEP 10] Saving Models")
    print("-" * 80)
    
    trainer.save_models(output_dir="model_pkls")
    
    # Save preprocessing objects
    import joblib
    joblib.dump(preprocessor.get_label_encoder(), "model_pkls/label_encoder.pkl")
    joblib.dump(preprocessor.get_scaler(), "model_pkls/scaler.pkl")
    print("✓ Preprocessing objects saved")
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  • Models: model_pkls/")
    print("  • Visualizations: logistic_regression/figure/")
    print("\nBest Model: Random Forest with threshold optimization")
    print(f"Optimal Threshold: {optimizer.optimal_threshold:.2f}")
    print(f"Expected Spoilage Recall: ~95%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
