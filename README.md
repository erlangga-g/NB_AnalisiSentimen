# Ordinal Rating Classification of Indonesian E-Commerce Reviews using Multinomial Naive Bayes

![Python Version](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nb-analisisentimen-skripsi.streamlit.app/)

🌐 **Live Demo:** [Open the Streamlit App](https://nb-analisisentimen-skripsi.streamlit.app/)

This repository contains the source code, trained models, and supporting resources developed as part of an undergraduate thesis on **ordinal rating classification (1–5 stars)** of Indonesian e-commerce reviews using Multinomial Naive Bayes. The proposed framework addresses severe class imbalance through SMOTE resampling and incorporates comprehensive NLP preprocessing, n-gram feature extraction, and hyperparameter optimization.

---

## 🌐 Web Application

The project includes an interactive Streamlit web application for predicting **ordinal ratings (1–5 stars)** from Indonesian e-commerce reviews.

### Home Page

<p align="center">
  <img src="images/streamlit-home.png" width="900" alt="Streamlit Home">
</p>

---

## 🚀 Core Features

- **Comprehensive Text Preprocessing:** Includes noise removal, text cleaning, case folding, tokenization, stopword removal, Indonesian stemming using the `Sastrawi` library, and slang normalization via a Custom Hybrid Dictionary (1,320 entry corpus).

- **Automated N-Gram Evaluation:** An automated evaluation to determine the most predictive feature range (Unigram, Bigram, and Trigram combinations).

- **Strict CV + SMOTE Data Isolation:** Class imbalance is handled using SMOTE inside an `ImbPipeline` to completely prevent data leakage during validation.

- **Hyperparameter Tuning:** Systematic optimization of Laplace/Lidstone smoothing (`α`) and prior fitting using `GridSearchCV` with Stratified 10-Fold Cross Validation.

---

## 📚 Data Sources

This project is built upon publicly available datasets and linguistic resources.

### Tokopedia Product Reviews Dataset

- Source: Farhan
- Original dataset containing **40,607** Indonesian product reviews collected from Tokopedia.
- This research uses a cleaned subset consisting of **37,146 unique reviews** after preprocessing and duplicate removal.
- Dataset: [https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews]

### Indonesian Slang Dictionary

- Source: sodolanangbjkatio
- Original public slang dictionary used as the baseline resource.
- Extended with a manually curated custom dictionary to create the Hybrid Slang Dictionary used throughout this study.
- Dataset: [https://www.kaggle.com/datasets/sodolanangbjkatio/slang-indonesia]

---

## 🙏 Acknowledgments

Special thanks to the authors of the publicly available Tokopedia Product Reviews dataset and the Indonesian Slang Dictionary on Kaggle. Their work made this research possible.

---

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
└── LICENSE
```

### Directory Overview

| Directory / File | Description |
|------------------|-------------|
| `app.py` | Streamlit web application for interactive rating prediction. |
| `data/raw/` | Original Tokopedia review dataset and slang dictionaries. |
| `data/processed/` | Intermediate datasets generated during preprocessing. |
| `images/` | Figures and screenshots used in the README and thesis. |
| `models/` | Serialized trained Multinomial Naive Bayes model. |
| `resources/` | Hybrid slang dictionary and preprocessing resources. |
| `notebooks/` | Main Jupyter Notebook containing the complete experimental pipeline. |

---

## 🛠️ Installation & Usage Guide

This project was developed using **Python 3.11**. The examples below use **Miniconda** for environment management, but any isolated Python environment (e.g., `venv` or `uv`) can be used.

### 1. Clone the Repository

```bash
git clone https://github.com/erlangga-g/NB_AnalisiSentimen.git
cd NB_AnalisiSentimen
```

### 2. Create and Activate a Miniconda Environment

```bash
conda create -n Skripsi-NaiveBayes python=3.11
conda activate Skripsi-NaiveBayes
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open:

```text
notebooks/NB_SentimentAnalysis_Skripsi_2.ipynb
```

Run the notebook sequentially from the first cell to reproduce the complete experimental pipeline.

> **Note**
>
> Ensure that the dataset and supporting dictionary files remain in their original locations inside the `data/` directory. The notebook uses relative paths to load these resources.

---

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
11. Performance Evaluation & Analysis

---

## 📈 Experimental Results

### Streamlit Prediction Example

<p align="center">
  <img src="images/hasil1-streamlit.png" width="900" alt="TF-IDF (Contribution Value)">
</p>

<p align="center">
  <img src="images/hasil2-streamlit.png" width="900" alt="Preprocessing details">
</p>

### Confusion Matrix

<p align="center">
  <img src="images/confusion_matrix_comparison_final.png" width="650" alt="Confusion Matrix">
</p>

### Performance Improvement Across Experimental Scenarios

<p align="center">
  <img src="images/ablation_study_5_scenarios.png" width="700" alt="Classification Report">
</p>

---

## 📈 Evaluation Metrics

The proposed framework is primarily evaluated using **Macro F1-Score**, supported by additional metrics including Accuracy, Precision, Recall, Confusion Matrix, and Stratified 10-Fold Cross Validation.

---

## 📚 Main Libraries

- scikit-learn
- imbalanced-learn
- pandas
- numpy
- nltk
- Sastrawi
- matplotlib
- joblib
- streamlit

---

## 📄 Citation

Citation information will be added after the undergraduate thesis has been officially completed and published.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.