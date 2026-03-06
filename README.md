# NLP-Projects
This repository contains **4 end-to-end NLP applications** developed during my internship. Each project demonstrates practical applications of Natural Language Processing, feature engineering, and model evaluation with interactive web interfaces.

---
## 📂 Repository Structure
NLP-Projects/
│
├── 📰 News Category Classification/
│ └── news_category_webapp.py
│
├── 🤖 Question Answering System with Transformers/
│ └── qa_system_webapp.py
│
├── 📰 Fake News Detection System/
│ └── fake_news_webapp.py
│
├── 🎬 Movie Reviews Sentiment Analyzer/
│ └── movie_sentiment_analysis_webapp.py
│
└── README.md

---
## 🧠 Question Answering System with Transformers

A transformer-based Question Answering (QA) system that generates precise answers from textual passages.

**Key Features:**
- Extracts answers in real-time from complex text passages.
- Uses **DistilBERT** for fast inference and **RoBERTa** for robust accuracy.
- Highlights answers in context with confidence scoring.
- Tracks question history and visualizes model performance metrics.

**Technologies Used:**
- Python, PyTorch
- Hugging Face Transformers
- Streamlit (web interface)
- Pandas, NumPy, Plotly

**Dataset:**  
[Stanford Question Answering Dataset (SQuAD v1.1) on Kaggle](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset)  

**Usage:**
1. Install required packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run qa_system_webapp.py`
3. Enter a passage and a question to get real-time answers.

---

## 📰 News Category Classification

A machine learning system to classify news articles into four categories: World, Sports, Business, and Sci/Tech.

**Key Features:**
- Preprocessing pipeline: tokenization, lemmatization, stopword removal.
- TF-IDF vectorization with n-grams for feature engineering.
- Model comparison: Logistic Regression, SVM, XGBoost (SVM achieved highest accuracy: 0.8144).
- Interactive metrics visualization including Confusion Matrix and Radar Charts via Streamlit.

**Technologies Used:**
- Python, Scikit-learn, XGBoost
- NLTK for NLP preprocessing
- Streamlit (web interface)
- Pandas, Matplotlib, Plotly

**Dataset:**  
[AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  

**Usage:**
1. Install required packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run news_category_webapp.py`
3. Upload or input news articles to get category predictions.

---

# Project 3: Fake News Detection System

A **Streamlit web app** that classifies news articles as **Real** or **Fake** using multiple NLP models and ensemble learning.

## Overview
The app processes news text and applies three models (**Naive Bayes, Logistic Regression, SVM**) to detect fake news in real-time. Predictions are combined using majority voting for higher accuracy.

## Key Features

- Real-time fake news detection  
- Ensemble learning with **3 models**  
- Confidence scoring for predictions  
- Word cloud visualization and performance dashboard  

## Model Performance

| Model | Accuracy | F1 Score |
|-------|---------|----------|
| Naive Bayes | ~94% | ~94% |
| Logistic Regression | ~95% | ~95% |
| SVM (TF-IDF) | ~95% | ~95% |

**Dataset:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) – 44,000+ articles  

## How to Run

```bash
cd "Fake News Detection System"
pip install -r requirements.txt
streamlit run fake_news_webapp.py
```

---

# Project 4: Movie Reviews Sentiment Analyzer

A **Streamlit web app** that classifies IMDB movie reviews as **Positive** or **Negative** using multiple machine learning models with ensemble voting.

## Overview
The app preprocesses review text and applies three classifiers (**Naive Bayes, Logistic Regression, SVM**) to predict sentiment in real-time. Ensemble learning improves prediction robustness.

## Key Features

- Real-time sentiment prediction  
- Ensemble of **3 models** with majority voting  
- Confidence scoring and word cloud visualizations  
- Review length analysis and performance dashboard  

## Model Performance

| Model | Accuracy | F1 Score |
|-------|---------|----------|
| Naive Bayes | ~85% | ~85% |
| Logistic Regression | ~89% | ~89% |
| SVM (TF-IDF) | ~89% | ~89% |

**Dataset:** [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) – 50,000 reviews  

## How to Run

```bash
cd "Movie Reviews Sentiment Analyzer"
pip install -r requirements.txt
streamlit run movie_sentiment_analysis_webapp.py
```
