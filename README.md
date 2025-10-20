# Detecting Sarcasm in Reddit Comments using Machine Learning

## Project Overview

This repository documents the end-to-end workflow for training a Logistic Regression classifier to detect sarcasm in Reddit comments. The dataset contains approximately 700,000 samples with over 5,000 features after preprocessing.

### Best Model Performance

**AUC = 0.7604** (5-fold cross-validation)  
**Model Type:** Logistic Regression (L2 penalty, C = 10, `max_iter = 2500`)

### Course: CS3244 Machine Learning

## Rpository Contents

- `Logistic_Regression_Model_Training.ipynb`: Main notebook outlining the full training, tuning, and evaluation workflow
- `models/`: Directory containing all saved `.pkl` model files from each training iteration

## Training Workflow

| Iteration | Objective | Key Outcome |
|-----------|-----------|-------------|
| 1 | Evaluate performance across different training data fractions | AUC plateaus after ~50–75% of data |
| 2 | Hyperparameter tuning on 10% subset | Best model found: L2 penalty, C = 10 |
| 3 | Train on full dataset (`max_iter = 1000`) | AUC = 0.7582 |
| 4 | Extend iterations to ensure convergence (`max_iter = 2500`) | Final AUC = 0.7604 |
