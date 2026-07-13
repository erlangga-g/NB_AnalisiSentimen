# Indonesian E-Commerce Sentiment Analysis using Multinomial Naive Bayes & SMOTE

![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

This repository contains the source code and data assets for an undergraduate thesis project focused on ordinal classification sentiment analysis (1-5 Star Ratings) of Indonesian e-commerce reviews. The system is heavily optimized to handle severely imbalanced datasets and enhanced using N-Gram feature extraction techniques.

## 🚀 Core Features
* **Comprehensive Text Preprocessing:** Includes noise removal, text cleaning, case folding, slang normalization via a Custom Hybrid Dictionary, and Indonesian stemming using the `Sastrawi` library.
* **Automated N-Gram Tournament:** An automated grid-like evaluation to determine the most predictive feature range (Unigram, Bigram, and Trigram).
* **Strict CV + SMOTE Data Isolation:** Class imbalance resolution via Synthetic Minority Over-sampling Technique (SMOTE), strictly isolated within an `ImbPipeline` to completely prevent any data leakage.
* **Hyperparameter Tuning:** Systematic optimization of *Laplace/Lidstone Smoothing* ($\alpha$) and class prior probabilities using `GridSearchCV` evaluated across 10-Fold Cross-Validation.

## 📁 Repository Structure
* `data_sample.csv` / `slang_indo.xls`: Main review dataset and the baseline slang dictionary from Kaggle.
* `notebook_skripsi.ipynb`: The main Jupyter Notebook executing the entire pipeline from Preprocessing to Evaluation.
* `requirements.txt`: Environment manifest tracking precise dependency versions.
* `*.png`: Exported high-resolution evaluation charts (Ablation Study, Alpha Sensitivity, and Confusion Matrix).

## 🛠️ Installation & Usage Guide
This project is built using Python 3.12 and managed via the ultra-fast `uv` package manager.

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Skripsi-NaiveBayes.git](https://github.com/YOUR_USERNAME/Skripsi-NaiveBayes.git)
   cd Skripsi-NaiveBayes