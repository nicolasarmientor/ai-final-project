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

## Dataset Summary

The core file `complete_raw_data.csv` contains the raw data from the chicken samples. The samples were exposed to different temperatures over multiple days.

For each sample the most important columns contain the VOCs captured from the machine, the microbial load (log CFU/g), and the freshness category for each measured day.

| Label    | Microbial Load (log CFU/g) |
|----------|-----------------------------|
| Fresh    | < 5.0                       |
| Moderate | 5.0 – 7.0                   |
| Spoiled  | ≥ 7.0                       |

## Implemented Machine-Learning Models

1. Linear Regression (LRg) - Microbial Load Prediction

Predicts microbial load using:
- Storage day
- VOC relevance index

Performance:
- R2 = 0.93
- RMSE = 0.47
- MAE = 0.37

2. Logistic Regression (LR) - Freshness Classification

Uses numeric VOC features:
- VOC count
- Relevance index
- Storage day
- Treatment type

Performance:
| Class               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Fresh               | 0.99      | 0.94   | 0.97     |
| Moderate            | 0.91      | 0.98   | 0.94     |
| Spoiled             | 1.00      | 0.98   | 0.99     |
| **Accuracy = 0.96** |           |        |          |

3. Multinomial Naïve Bayes (MNB) - VOC Identity Classification

Uses VOC presence/absense as categorical inputs.

Performance:
| Class               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Fresh               | 0.71      | 0.50   | 0.59     |
| Moderate            | 0.48      | 0.77   | 0.59     |
| Spoiled             | 1.00      | 0.75   | 0.86     |
| **Accuracy = 0.63** |           |        |          |

4. Random Forest (RFC) - Biomarker-Based Freshness Estimation
Trained using 18 curated VOC compounds.

Performance (R1 test set):
- Accuracy: **56.6%**
- Spoiled recall: **0.75**

## How to Run the Project
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run Linear Regression Model
Training:
```base
python linear_regression/linear_regression_train.py
```
Evaluation:
```bash
python linear_regression/linear_regression_results.py
```
Inference:
```bash
python linear_regression/linear_regression_infer.py
```

3. Run Logistic Regression Model
Training:
```bash
python logistic_regression/logistic_regression_train.py
```
Evaluation:
```bash
python logistic_regression/logistic_regression_results.py
```
Inference:
```bash
python logistic_regression/logistic_regression_infer.py
```

4. Run Naïve Bayes Model
Training:
```bash
python multinomial_naive_bayes/naive_bayes_train.py
```
Evaluation:
```bash
python multinomial_naive_bayes/naive_bayes_results.py
```
Inference:
```bash
python multinomial_naive_bayes/naive_bayes_infer.py
```

5. Run Random Forest Model
Execution:
```bash
python categorization/main.py
```

## Contributors
| Name                  | Role                          |
| --------------------- | ----------------------------- |
| **Vianca Tashiguano** | Data collection, reporting    |
| **Josué Céspedes**    | Modeling, visualization       |
| **Nicolás Sarmiento** | Code implementation, modeling |

