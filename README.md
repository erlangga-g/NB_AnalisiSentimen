# Indonesian E-Commerce Sentiment Analysis Using Multinomial Naive Bayes and SMOTE Resampling Optimized with n-gram Features and Hyperparameter Tuning

![Python Version](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

This repository contains the source code and data assets for an undergraduate thesis project focused on ordinal classification sentiment analysis (1-5 Star Ratings) of Indonesian e-commerce reviews. The system is heavily optimized to handle severely imbalanced datasets and enhanced using n-gram feature extraction techniques.

## 🚀 Core Features
* **Comprehensive Text Preprocessing:** Includes noise removal, text cleaning, case folding, tokenization, stopword removal, Indonesian stemming using the `Sastrawi` library, and slang normalization via a Custom Hybrid Dictionary (1,320 entry corpus).
* **Automated n-gram Tournament:** An automated grid-like evaluation to determine the most predictive feature range (Unigram, Bigram, and Trigram combinations).
* **Strict CV + SMOTE Data Isolation:** Class imbalance resolution via Synthetic Minority Over-sampling Technique (SMOTE), strictly isolated within an `ImbPipeline` framework to completely prevent any data leakage during validation.
* **Hyperparameter Tuning:** Systematic optimization of *Laplace/Lidstone Smoothing* ($\alpha$) and prior fitting configurations using `GridSearchCV` evaluated across Stratified 10-Fold Cross-Validation.

## 📁 Repository Structure
* `data_sample.csv`: Main Tokopedia review dataset containing 37,146 clean unique records.
* `slang_indo.xls` / `kamus_kustom.txt`: Baseline Kaggle dictionary and the locally engineered 116-entry patch dictionary used to form the Hybrid corpus.
* `NB_SentimentAnalysis_Skripsi_2.ipynb`: The main Jupyter Notebook executing the entire pipeline from Preprocessing to evaluation metrics.
* `requirements.txt`: Environment manifest tracking precise dependency versions for replication.

## 🛠️ Installation & Usage Guide
This project is built using Python 3.11 and managed within a isolated environment using **Miniconda**.

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/erlangga-g/NB_AnalisiSentimen.git](https://github.com/erlangga-g/NB_AnalisiSentimen.git)
   cd NB_AnalisiSentimen