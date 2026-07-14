# Ordinal Rating Classification of Indonesian E-Commerce Reviews using Multinomial Naive Bayes

![Python Version](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nb-analisisentimen-skripsi.streamlit.app/)

🌐 **Live Demo:** https://nb-analisisentimen-skripsi.streamlit.app/

This repository contains the source code and supporting data for an undergraduate thesis on **ordinal rating classification (1–5 stars)** of Indonesian e-commerce reviews using Multinomial Naive Bayes. The proposed framework addresses severe class imbalance through SMOTE resampling and incorporates comprehensive NLP preprocessing, n-gram feature extraction, and hyperparameter optimization.

## 🚀 Core Features
* **Comprehensive Text Preprocessing:** Includes noise removal, text cleaning, case folding, tokenization, stopword removal, Indonesian stemming using the `Sastrawi` library, and slang normalization via a Custom Hybrid Dictionary (1,320 entry corpus).
* **Automated n-gram Tournament:** An automated grid-like evaluation to determine the most predictive feature range (Unigram, Bigram, and Trigram combinations).
* **Strict CV + SMOTE Data Isolation:** Class imbalance resolution via Synthetic Minority Over-sampling Technique (SMOTE), strictly isolated within an `ImbPipeline` framework to completely prevent any data leakage during validation.
* **Hyperparameter Tuning:** Systematic optimization of *Laplace/Lidstone Smoothing* ($\alpha$) and prior fitting configurations using `GridSearchCV` evaluated across Stratified 10-Fold Cross-Validation.

## 📁 Repository Structure

```text
NB_AnalisiSentimen/
├── app.py                              # Streamlit web application
├── data/
│   ├── raw/                            # Original dataset and slang dictionaries
│   └── processed/                      # Processed datasets
├── images/                             # Figures and visualizations
├── models/
│   └── model_naive_bayes_skripsi.pkl   # Trained Multinomial Naive Bayes model
├── resources/
│   └── kamus_slang_hybrid.pkl          # Hybrid slang dictionary
├── notebooks/
│   └── NB_SentimentAnalysis_Skripsi_2.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

### Directory Overview

| Directory / File | Description |
|------------------|-------------|
| `app.py` | Streamlit web application for interactive sentiment rating prediction. |
| `data/raw/` | Original Tokopedia review dataset and slang dictionaries. |
| `data/processed/` | Intermediate datasets generated during preprocessing. |
| `images/` | Figures and visualizations used in the README and thesis. |
| `models/` | Serialized NLP resources, including the trained Multinomial Naive Bayes model. |
| `resources/` | Hybrid slang dictionary and other NLP resources used during preprocessing. |
| `notebooks/` | Main Jupyter Notebook containing the complete experimental pipeline. |

## 🛠️ Installation & Usage Guide

This project was developed using **Python 3.11**. The examples below use **Miniconda** for environment management, but any isolated Python environment (e.g., `venv` or `uv`) can be used.

### 1. Clone the Repository

```bash
git clone https://github.com/erlangga-g/NB_AnalisiSentimen.git
cd NB_AnalisiSentimen
```

### 2. Create and Activate a Minionda Environment

```bash
conda create -n Skripsi-NaiveBayes python=3.11
conda activate Skripsi-NaiveBayes
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open:

```
notebooks/NB_SentimentAnalysis_Skripsi_2.ipynb
```

Run the notebook sequentially from the first cell to reproduce the complete experimental pipeline.

> **Note**
>
> Ensure that the dataset and supporting dictionary files remain in their original locations inside the `data/` directory. The notebook uses relative paths to load these resources.


## 📊 Experimental Pipeline

The complete workflow consists of the following stages:

1. Dataset Loading
2. Text Cleaning
3. Slang Normalization
4. Tokenization
5. Stopword Removal
6. Indonesian Stemming (Sastrawi)
7. TF-IDF Vectorization with n-Gram Features
8. SMOTE Resampling
9. Hyperparameter Optimization (GridSearchCV)
10. Multinomial Naive Bayes Training
11. Performance Evaluation

## 📈 Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Stratified 10-Fold Cross Validation

## 📚 Main Libraries

- scikit-learn
- imbalanced-learn
- pandas
- numpy
- nltk
- Sastrawi
- matplotlib

## 📄 Citation

Citation information will be added after the undergraduate thesis has been officially completed and published.