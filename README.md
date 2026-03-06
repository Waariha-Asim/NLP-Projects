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

A machine learning application that analyzes news articles and classifies them as **Real** or **Fake** using Natural Language Processing and multiple classification models.

## Overview
This system detects misleading or fake news articles by processing text through an NLP pipeline and applying ensemble learning to produce the final prediction.

## Machine Learning Models

| Model | Description | Vectorization |
|------|-------------|---------------|
| Naive Bayes | Probabilistic classifier based on Bayes' theorem | Count Vectorizer |
| Logistic Regression | Linear model for binary classification | Count Vectorizer |
| Support Vector Machine (SVM) | Maximum-margin classifier with a linear kernel | TF-IDF Vectorizer |

**Ensemble Method:**  
Final predictions are generated using **majority voting** from all three models.

## NLP Pipeline

Raw Text → Lowercasing → Special Character Removal → Stopword Removal → Lemmatization → Vectorization

## Features

- Real-time fake news detection
- Ensemble prediction using three ML models
- Confidence score for predictions
- Word cloud visualization comparing Fake vs Real news
- Model performance comparison dashboard
- Sample articles for testing
- Confusion matrix visualization
- Radar charts for multi-metric comparison
- Modern dark UI with glassmorphism design

## Dataset

Dataset: **Fake and Real News Dataset**

Kaggle Link:  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Details:
- 44,000+ news articles
- Classes: Fake (0) and Real (1)
- Files used: `Fake.csv` and `True.csv`
- Feature used: Combined **title and article text**

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|----------|-----------|--------|----------|--------|
| Naive Bayes | ~94% | ~94% | ~94% | ~94% | ~94% |
| Logistic Regression | ~95% | ~95% | ~95% | ~95% | ~95% |
| SVM (TF-IDF) | ~95% | ~95% | ~95% | ~95% | ~95% |

## Technologies Used

- Streamlit
- Scikit-learn
- NLTK
- Pandas and NumPy
- Plotly, Matplotlib, Seaborn
- WordCloud
- Custom CSS for UI design

## How to Run

```bash
cd "Fake News Detection System"
pip install streamlit pandas numpy matplotlib seaborn wordcloud plotly nltk scikit-learn
streamlit run fake_news_webapp.py
```

---
# Project 4: Movie Reviews Sentiment Analyzer

A machine learning application that analyzes movie reviews and classifies them as **Positive** or **Negative** using Natural Language Processing techniques.

## Overview
This system performs sentiment analysis on IMDB movie reviews using multiple machine learning models combined with ensemble learning to improve prediction accuracy.

## Machine Learning Models

| Model | Description | Vectorization |
|------|-------------|---------------|
| Naive Bayes | Probabilistic classifier suitable for text classification | Count Vectorizer |
| Logistic Regression | Linear model with high interpretability | Count Vectorizer |
| Support Vector Machine (SVM) | Maximum-margin classifier with a linear kernel | TF-IDF Vectorizer |

**Ensemble Method:**  
Final sentiment prediction is generated using **majority voting across all three models**.

## NLP Pipeline

HTML Removal → Lowercasing → Special Character Removal → Stopword Removal → Lemmatization → Vectorization

## Features

- Real-time sentiment analysis
- Ensemble prediction using three ML models
- Confidence score for predictions
- Word cloud visualization for Positive vs Negative reviews
- Review length distribution analysis
- Sample reviews for testing
- Radar charts for model comparison
- Interactive performance dashboard
- Modern full-width UI with navigation tabs

## Dataset

Dataset: **IMDB Movie Reviews Dataset**

Kaggle Link:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Details:
- 50,000 movie reviews
- 25,000 positive reviews
- 25,000 negative reviews
- File used: `IMDB Dataset.csv`
- Feature: Raw review text with HTML tags

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|----------|-----------|--------|----------|--------|
| Naive Bayes | ~85% | ~85% | ~85% | ~85% | ~85% |
| Logistic Regression | ~89% | ~89% | ~89% | ~89% | ~89% |
| SVM (TF-IDF) | ~89% | ~89% | ~89% | ~89% | ~89% |

## Technologies Used

- Streamlit
- Scikit-learn
- NLTK
- Pandas and NumPy
- Plotly, Matplotlib, Seaborn
- WordCloud
- Custom CSS for UI design

## How to Run

```bash
cd "Movie Reviews Sentiment Analyzer"
pip install streamlit pandas numpy matplotlib seaborn wordcloud plotly nltk scikit-learn
streamlit run movie_sentiment_analysis_webapp.py
```

