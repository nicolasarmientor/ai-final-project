"""
Master Pipeline - All Models

Orchestrates execution of:
1. Categorization (Random Forest with VOC features)
2. Logistic Regression
3. Multinomial Naive Bayes
"""

import sys
import os
from pathlib import Path
import subprocess

# Project root
PROJECT_ROOT = Path(__file__).parent


def run_categorization():
    """Run categorization model (Random Forest)"""
    
    print("\n" + "="*80)
    print("EXECUTING: CATEGORIZATION (RANDOM FOREST + VOC FEATURES)")
    print("="*80)
    
    try:
        categorization_main = PROJECT_ROOT / "categorization" / "main.py"
        result = subprocess.run(
            [sys.executable, str(categorization_main)],
            cwd=str(categorization_main.parent),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n[OK] Categorization completed successfully")
            return True
        else:
            print(f"\n[ERROR] Categorization failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Categorization execution failed: {e}")
        return False


def run_logistic_regression():
    """Run logistic regression model"""
    
    print("\n" + "="*80)
    print("EXECUTING: LOGISTIC REGRESSION")
    print("="*80)
    
    try:
        lr_dir = PROJECT_ROOT / "logistic_regression"
        
        # Train
        print("\n[STEP 1] Training Logistic Regression...")
        lr_train = lr_dir / "logistic_regression_train.py"
        result = subprocess.run(
            [sys.executable, str(lr_train)],
            cwd=str(PROJECT_ROOT),  # Run from project root, not lr_dir
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Training failed with return code {result.returncode}")
            return False
        
        # Results
        print("\n[STEP 2] Generating Logistic Regression results...")
        lr_results = lr_dir / "logistic_regression_results.py"
        result = subprocess.run(
            [sys.executable, str(lr_results)],
            cwd=str(PROJECT_ROOT),  # Run from project root, not lr_dir
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n[OK] Logistic Regression completed successfully")
            return True
        else:
            print(f"[ERROR] Results generation failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Logistic Regression execution failed: {e}")
        return False


def run_naive_bayes():
    """Run multinomial Naive Bayes model"""
    
    print("\n" + "="*80)
    print("EXECUTING: MULTINOMIAL NAIVE BAYES")
    print("="*80)
    
    try:
        nb_dir = PROJECT_ROOT / "multinomial_naive_bayes"
        
        # Train
        print("\n[STEP 1] Training Naive Bayes...")
        nb_train = nb_dir / "naive_bayes_train.py"
        result = subprocess.run(
            [sys.executable, str(nb_train)],
            cwd=str(PROJECT_ROOT),  # Run from project root, not nb_dir
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Training failed with return code {result.returncode}")
            return False
        
        # Results
        print("\n[STEP 2] Generating Naive Bayes results...")
        nb_results = nb_dir / "naive_bayes_results.py"
        result = subprocess.run(
            [sys.executable, str(nb_results)],
            cwd=str(PROJECT_ROOT),  # Run from project root, not nb_dir
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n[OK] Naive Bayes completed successfully")
            return True
        else:
            print(f"[ERROR] Results generation failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Naive Bayes execution failed: {e}")
        return False


def main():
    """Execute all models"""
    
    print("\n" + "="*80)
    print("MASTER PIPELINE - ALL MODELS")
    print("="*80)
    print(f"""
This master pipeline will execute:
  1. Categorization (Random Forest with VOC features)
     - Includes extended validation on R2 and R3
     - Output: categorization/figure/ + categorization/extended_validation/figures/
  
  2. Logistic Regression
     - Output: logistic_regression/figure/
  
  3. Multinomial Naive Bayes
     - Output: multinomial_naive_bayes/figure/

All models use the same 70-15-15 train-validation-test split.
""")
    
    results = {}
    
    # Run all models
    print("\n" + "="*80)
    print("STARTING PIPELINE EXECUTION")
    print("="*80)
    
    results['categorization'] = run_categorization()
    results['logistic_regression'] = run_logistic_regression()
    results['naive_bayes'] = run_naive_bayes()
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    print("\nModel Status:")
    print(f"  Categorization:        {'[OK] SUCCESS' if results['categorization'] else '[ERROR] FAILED'}")
    print(f"  Logistic Regression:   {'[OK] SUCCESS' if results['logistic_regression'] else '[ERROR] FAILED'}")
    print(f"  Naive Bayes:           {'[OK] SUCCESS' if results['naive_bayes'] else '[ERROR] FAILED'}")
    
    all_success = all(results.values())
    
    print(f"\nOverall Status: {'[OK] ALL MODELS COMPLETE' if all_success else '[WARNING] SOME MODELS FAILED'}")
    
    print("""
Output Locations:
  - Categorization:      categorization/figure/ + categorization/extended_validation/figures/
  - Logistic Regression: logistic_regression/figure/
  - Naive Bayes:         multinomial_naive_bayes/figure/
  - Models:              model_pkls/

Next Steps:
  - Review figures in each model folder
  - Compare model performances
  - Check accuracy, precision, recall metrics
""")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
