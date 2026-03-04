# NLP-Projects
This repository contains end-to-end NLP applications developed during my internship at **ElevvoPathways**. Both projects demonstrate practical applications of Natural Language Processing, feature engineering, and model evaluation with interactive web interfaces.

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

**Project Structure:**

Question Answering System with Transformers/
└── qa_system_webapp.py


**Dataset:**  
[Stanford Question Answering Dataset (SQuAD v1.1) on Kaggle](https://www.kaggle.com/datasets/rajyellow46/wikipedia-qa)  

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

**Project Structure:**

News Category Classification/
└── news_category_webapp.py


**Dataset:**  
[AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  

**Usage:**
1. Install required packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run news_category_webapp.py`
3. Upload or input news articles to get category predictions.
