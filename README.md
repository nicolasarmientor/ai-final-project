# 🧪 Non-Destructive Prediction of Chicken Freshness Using VOC Profiles
## Machine Learning Models for Microbial Load Estimation and Spoilage Classification

This repository containts the full implementation of a machine-learning pipeline designed to **predict microbial freshness of raw chicken** using **Volatile Organic Compound (VOC)** profiles collected through an **electronic-nose (E-nose) system**.

The methodology, models, and results are described in detail in the final project report.

## Project Overview

In fresh poultry, spoilage is driven primarily by microbial growth. Traditional microbial testing is accurate but slow, destructive, and impractical for real-time testing.

The aim of this project is to introduce a non-destructive approach that uses the following as methods:

- VOC profiles  
- Microbial load measurements  
- Supervised machine-learning models  

to:

1. Compute a microbial load estimation  
2. Classify freshness into three classes: Fresh, Moderate, Spoiled  

The models included in this repository make use of quantitative VOC features, compound identities, and metadata to train and evaluate precise and accurate results in prediction and classification.

## Repository Structure
```
ai-final-project/
├── .gitignore
├── main.py
├── requirements.txt
│
├── data/
│   ├── raw_data/
│   │   ├── DataAI.csv
│   │   └── complete_raw_data.csv
│   └── processed_data/
│       ├── linear_regression_data.csv
│       ├── logistic_regression_data.csv
│       └── naive_bayes_data.csv
│
├── model_pkls/
│   ├── categorization_model.pkl
│   ├── linear_regression_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_mlb.pkl
│   └── naive_bayes_model.pkl
│
├── linear_regression/
│   ├── linear_regression_train.py
│   ├── linear_regression_results.py
│   ├── linear_regression_infer.py
│   └── figure/
│       └── linear_regression_plot.png
│
├── logistic_regression/
│   ├── logistic_regression_train.py
│   ├── logistic_regression_results.py
│   ├── logistic_regression_infer.py
│   └── figure/
│       └── lr_confusion_matrix_heatmap.png
│
├── multinomial_naive_bayes/
│   ├── README.md
│   ├── naive_bayes_train.py
│   ├── naive_bayes_results.py
│   ├── naive_bayes_infer.py
│   └── figure/
│       └── nb_confusion_matrix_heatmap.png
│
└── categorization/
    ├── categorization.py      
    ├── check_split.py
    ├── visualization.py
    ├── main.py                    
    ├── figure/
    │   ├── 01_class_distribution.png
    │   ├── 02_confusion_matrices.png
    │   ├── 03_feature_importance.png
    │   └── 04_threshold_optimization.png
    └── extended_validation/
        ├── __init__.py
        ├── extended_validator.py
        ├── visualization.py
        └── figures/
            ├── class_distribution.png
            ├── confusion_matrix.png
            ├── metrics_comparison.png
            └── recall_by_class.png
```
