# **SpamX – Intelligent Spam Email Classifier (Binary Classification)**

## Overview

**SpamX** is a Machine Learning-based text classification system that detects whether a message is **Spam (1)** or **Not Spam (0)**.
It uses **NLP (Natural Language Processing)** techniques with **TF-IDF vectorization** and multiple machine learning models such as **Naïve Bayes, Logistic Regression, and SVM** to ensure accurate classification.

---

## Features

Text preprocessing (cleaning, stopword removal, tokenization)
TF-IDF vectorization for feature extraction
Multiple ML models trained & evaluated
Achieved **100% accuracy** on the given dataset
Extendable to real-world datasets for production use

---

## Dataset

* **Source:** [SMS Spam Collection Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
* **Size:** 5,572 messages
* **Labels:**

  * `ham` → Not Spam (0)
  * `spam` → Spam (1)

---

## Tech Stack

* **Python** 
* **Pandas & NumPy** → Data Processing
* **NLTK** → NLP (stopwords, tokenization, cleaning)
* **Scikit-learn** → ML Models & Evaluation
* **Matplotlib & Seaborn** → Visualization

---

## Workflow

1. **Data Preprocessing**

   * Lowercasing, removing punctuation & numbers
   * Removing stopwords (NLTK)
   * Converting text → vectors with **TF-IDF**

2. **Model Training**

   * **Naïve Bayes (MultinomialNB)**
   * **Logistic Regression**
   * **Support Vector Machine (SVM)**

3. **Model Evaluation**

   * Accuracy 
   * Precision 
   * Recall 
   * F1-Score 

---

## Results

| Model                  | Accuracy | Precision | Recall | F1-Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| Naïve Bayes            | 100%     | 1.00      | 1.00   | 1.00     |
| Logistic Regression    | 100%     | 1.00      | 1.00   | 1.00     |
| Support Vector Machine | 100%     | 1.00      | 1.00   | 1.00     |

**Note:** Perfect accuracy suggests the dataset may be too clean or overfitting. Testing with larger real-world datasets is recommended.

---

## How to Run

```bash
# Clone repo
git clone https://github.com/Aishvariyaa/SpamShield.git
cd SpamX

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook SMS Spam.ipynb
```

---

## Future Enhancements

Test with large real-world email datasets
Implement Deep Learning models (LSTM, BERT)
Deploy as a **Flask/FastAPI service** for real-time filtering
Build a simple **web app** for users to test messages

---

## Tagline
"SpamX -  Smart AI-powered email spam detection system"

---


