# Ordinal Rating Classification of Indonesian E-Commerce Reviews using Multinomial Naive Bayes

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-orange)

**Live Demo:** https://nb-analisisentimen-skripsi.streamlit.app/

This repository contains the implementation of an undergraduate thesis on **ordinal rating classification (1–5 stars)** for Indonesian e-commerce reviews using **Multinomial Naive Bayes**. The proposed framework combines comprehensive NLP preprocessing, TF-IDF with n-gram features, SMOTE resampling, and hyperparameter optimization to improve classification performance on imbalanced data.

<p align="center">
    <img src="images/streamlit-home.png" width="900" alt="Streamlit Home">
</p>

---

## Features

- Complete Indonesian NLP preprocessing pipeline
- Hybrid slang normalization (1,320 entries)
- TF-IDF with automated n-gram evaluation
- SMOTE resampling with leakage-free `ImbPipeline`
- Hyperparameter tuning using GridSearchCV
- Interactive Streamlit web application

---

## Repository Structure

```text
NB_AnalisiSentimen/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── images/
├── models/
├── notebooks/
├── resources/
├── requirements.txt
├── README.md
└── LICENSE
```

| Directory | Description |
|-----------|-------------|
| `app.py` | Streamlit application |
| `data/` | Dataset and preprocessing outputs |
| `images/` | README figures |
| `models/` | Trained model |
| `resources/` | Hybrid slang dictionary |
| `notebooks/` | Main experiment notebook |

---

## Installation

```bash
git clone https://github.com/erlangga-g/NB_AnalisiSentimen.git
cd NB_AnalisiSentimen
```

Create an isolated Python environment.

Example using Miniconda:

```bash
conda create -n Skripsi-NaiveBayes python=3.11
conda activate Skripsi-NaiveBayes
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Launch Jupyter.

```bash
jupyter lab
```

Open

```
notebooks/NB_SentimentAnalysis_Skripsi_2.ipynb
```

---

## Experimental Pipeline

```text
Dataset
   ↓
Cleaning
   ↓
Slang Normalization
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming
   ↓
TF-IDF + N-Gram
   ↓
SMOTE
   ↓
GridSearchCV
   ↓
Multinomial Naive Bayes
   ↓
Evaluation
```

---

## Experimental Results

### Streamlit Prediction

<p align="center">
<img src="images/hasil1-streamlit.png" width="900">
</p>

<p align="center">
<img src="images/hasil2-streamlit.png" width="900">
</p>

### Confusion Matrix

<p align="center">
<img src="images/confusion_matrix_comparison_final.png" width="650">
</p>

### Ablation Study

<p align="center">
<img src="images/ablation_study_5_scenarios.png" width="700">
</p>

The proposed framework is primarily evaluated using **Macro F1-Score**, supported by Accuracy, Precision, Recall, Confusion Matrix, and Stratified 10-Fold Cross Validation.

---

## Dataset & Resources

| Resource | Description |
|----------|-------------|
| Tokopedia Product Reviews | Original dataset: 40,607 reviews (37,146 used after cleaning) |
| Indonesian Slang Dictionary | Public Kaggle dictionary extended with a custom hybrid dictionary |

- Dataset: https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews
- Slang Dictionary: https://www.kaggle.com/datasets/sodolanangbjkatio/slang-indonesia

---

## Main Libraries

- scikit-learn
- imbalanced-learn
- pandas
- NumPy
- NLTK
- Sastrawi
- Matplotlib
- Joblib
- Streamlit

---

## Acknowledgments

This project builds upon publicly available datasets and linguistic resources from Kaggle. Sincere thanks to the original authors for making these resources openly available.

---

## Citation

Citation information will be added after the undergraduate thesis has been officially completed and published.

---

## License

This project is licensed under the MIT License.