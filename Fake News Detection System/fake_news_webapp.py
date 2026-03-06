import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Black Modern Theme (LinkedIn-inspired)
st.markdown("""
<style>
    /* Main background - Pure black with subtle texture */
    .stApp {
        background: #000000;
        background-image: 
            radial-gradient(at 40% 20%, rgba(138, 43, 226, 0.05) 0px, transparent 50%),
            radial-gradient(at 80% 0%, rgba(138, 43, 226, 0.03) 0px, transparent 50%),
            radial-gradient(at 0% 50%, rgba(138, 43, 226, 0.03) 0px, transparent 50%);
    }
    
    /* Headers - Professional white with purple accent */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 0 20px rgba(138, 43, 226, 0.3);
    }
    
    h2, h3 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px !important;
    }
    
    /* Sidebar - Dark purple gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0033 0%, #2d1b4e 50%, #1a0033 100%);
        border-right: 1px solid rgba(138, 43, 226, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(138, 43, 226, 0.5);
    }
    
    /* Buttons - Purple gradient with hover effects */
    .stButton>button {
        background: linear-gradient(135deg, #7b2cbf 0%, #9d4edd 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #9d4edd 0%, #c77dff 100%);
        box-shadow: 0 6px 25px rgba(138, 43, 226, 0.5);
        transform: translateY(-2px);
    }
    
    /* Text input - Dark with purple accent */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #3d3d3d;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, 
    .stTextArea>div>div>textarea:focus {
        border-color: #9d4edd;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.3);
    }
    
    /* Metrics - Purple accent */
    [data-testid="stMetricValue"] {
        color: #c77dff !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(199, 125, 255, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #9d4edd !important;
    }
    
    /* Tabs - Modern LinkedIn style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0d0d0d;
        border-radius: 12px;
        padding: 0.7rem;
        border: 1px solid #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #b0b0b0;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(138, 43, 226, 0.1);
        color: #c77dff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7b2cbf 0%, #9d4edd 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(138, 43, 226, 0.4);
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(138, 43, 226, 0.08);
        border-left: 4px solid #9d4edd;
        border-radius: 10px;
        color: #e0e0e0;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio > label, .stCheckbox > label {
        color: #e0e0e0 !important;
    }
    
    /* Dataframe - Dark theme */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #3d3d3d !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(138, 43, 226, 0.1);
        border-radius: 10px;
        color: #e0e0e0 !important;
        font-weight: 600;
        border: 1px solid rgba(138, 43, 226, 0.2);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(138, 43, 226, 0.15);
        border-color: #9d4edd;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #25a867;
        box-shadow: 0 4px 15px rgba(37, 168, 103, 0.2);
    }
    
    /* Error message */
    .error-box {
        background: linear-gradient(135deg, #721c24 0%, #a94442 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7b2cbf 0%, #9d4edd 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #9d4edd 0%, #c77dff 100%);
    }
    
    /* Cards and containers */
    .element-container {
        color: #e0e0e0;
    }
    
    /* Links */
    a {
        color: #c77dff !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #e0aaff !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7b2cbf 0%, #9d4edd 100%);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Preprocessing function
@st.cache_data
def preprocess_text(text):
    """Clean and preprocess text data"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters
    import re
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(sample_size=5000):
    """Load and preprocess the fake news dataset
    
    Args:
        sample_size: Number of articles to use for training (default: 5000)
    """
    try:
        with st.spinner('📊 Loading datasets...'):
            df_fake = pd.read_csv(r'C:\Users\AR FAST\Documents\NLP Internship Projects\Fake News Detection Project\Fake.csv')
            df_real = pd.read_csv(r'C:\Users\AR FAST\Documents\NLP Internship Projects\Fake News Detection Project\True.csv')
            
            df_fake['label'] = 0
            df_real['label'] = 1
            
            # Sample equal amounts from each class
            samples_per_class = sample_size // 2
            df_fake_sample = df_fake.sample(n=min(samples_per_class, len(df_fake)), random_state=42)
            df_real_sample = df_real.sample(n=min(samples_per_class, len(df_real)), random_state=42)
            
            df = pd.concat([df_fake_sample, df_real_sample], axis=0)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        with st.spinner('🔄 Preprocessing text data...'):
            df['text'] = df['title'] + " " + df['text']
            df['text'] = df['text'].apply(preprocess_text)
            df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Train models
@st.cache_resource
def train_models(df):
    """Train all classification models"""
    X = df['text']
    y = df['label']
    
    # CountVectorizer - reduced features for faster training
    vectorizer_count = CountVectorizer(max_features=3000)
    X_counts = vectorizer_count.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=42)
    
    # TF-IDF - reduced features for faster training
    vectorizer_tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = vectorizer_tfidf.fit_transform(X)
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
        X_tfidf, y, test_size=0.3, random_state=42
    )
    
    models = {}
    results = []
    
    with st.spinner('🤖 Training Naive Bayes...'):
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        models['Naive Bayes'] = nb_model
        results.append({
            'Model': 'Naive Bayes',
            'Accuracy': accuracy_score(y_test, nb_pred),
            'Precision': precision_score(y_test, nb_pred),
            'Recall': recall_score(y_test, nb_pred),
            'F1-Score': f1_score(y_test, nb_pred),
            'ROC-AUC': roc_auc_score(y_test, nb_pred)
        })
    
    with st.spinner('🤖 Training Logistic Regression...'):
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        models['Logistic Regression'] = lr_model
        results.append({
            'Model': 'Logistic Regression',
            'Accuracy': accuracy_score(y_test, lr_pred),
            'Precision': precision_score(y_test, lr_pred),
            'Recall': recall_score(y_test, lr_pred),
            'F1-Score': f1_score(y_test, lr_pred),
            'ROC-AUC': roc_auc_score(y_test, lr_pred)
        })
    
    with st.spinner('🤖 Training SVM...'):
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train_tfidf, y_train_tfidf)
        svm_pred = svm_model.predict(X_test_tfidf)
        models['SVM (TF-IDF)'] = svm_model
        results.append({
            'Model': 'SVM (TF-IDF)',
            'Accuracy': accuracy_score(y_test_tfidf, svm_pred),
            'Precision': precision_score(y_test_tfidf, svm_pred),
            'Recall': recall_score(y_test_tfidf, svm_pred),
            'F1-Score': f1_score(y_test_tfidf, svm_pred),
            'ROC-AUC': roc_auc_score(y_test_tfidf, svm_pred)
        })
    
    return models, pd.DataFrame(results), vectorizer_count, vectorizer_tfidf

# Sidebar
with st.sidebar:
    st.markdown("# 📰 Fake News Detector")
    st.markdown("---")
    st.markdown("### 🎯 Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🔮 Predict", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Sample size selector
    sample_size = st.selectbox(
        "Training Sample Size",
        options=[2000, 5000, 10000, 15000],
        index=1,
        help="Smaller samples train faster (recommended: 5000)"
    )
    
    st.info(f"⚡ Using {sample_size:,} articles (~{sample_size//2:,} fake + ~{sample_size//2:,} real)")
    
    if st.button("🔄 Train/Reload Models"):
        with st.spinner("Training models..."):
            df = load_and_preprocess_data(sample_size=sample_size)
            if df is not None:
                st.session_state.df = df
                st.session_state.sample_size = sample_size
                models, results, vec_count, vec_tfidf = train_models(df)
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.vectorizer_count = vec_count
                st.session_state.vectorizer_tfidf = vec_tfidf
                st.session_state.models_trained = True
                st.success("✅ Models trained successfully!")
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if st.session_state.df is not None:
        df = st.session_state.df
        sample_size = st.session_state.get('sample_size', len(df))
        st.metric("Sample Size", f"{sample_size:,}")
        st.metric("Total Articles", f"{len(df):,}")
        st.metric("Fake News", f"{(df['label']==0).sum():,}")
        st.metric("Real News", f"{(df['label']==1).sum():,}")

# Main content
if page == "🏠 Home":
    st.markdown("# 📰 Fake News Detection System")
    st.markdown("### *AI-Powered News Authenticity Analyzer*")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4a148c 0%, #6a1b9a 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;
                    border: 1px solid rgba(138, 43, 226, 0.3);
                    box-shadow: 0 8px 20px rgba(138, 43, 226, 0.3);'>
            <h2 style='color: white; margin: 0; font-size: 3rem;'>🤖</h2>
            <h3 style='color: white; margin: 0.5rem 0; font-weight: 700;'>3 ML Models</h3>
            <p style='color: #e1bee7; margin: 0; font-weight: 500;'>Ensemble prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #6a1b9a 0%, #8e24aa 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;
                    border: 1px solid rgba(138, 43, 226, 0.3);
                    box-shadow: 0 8px 20px rgba(138, 43, 226, 0.3);'>
            <h2 style='color: white; margin: 0; font-size: 3rem;'>⚡</h2>
            <h3 style='color: white; margin: 0.5rem 0; font-weight: 700;'>Real-time</h3>
            <p style='color: #e1bee7; margin: 0; font-weight: 500;'>Instant analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #8e24aa 0%, #9c27b0 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;
                    border: 1px solid rgba(138, 43, 226, 0.3);
                    box-shadow: 0 8px 20px rgba(138, 43, 226, 0.3);'>
            <h2 style='color: white; margin: 0; font-size: 3rem;'>📊</h2>
            <h3 style='color: white; margin: 0.5rem 0; font-weight: 700;'>High Accuracy</h3>
            <p style='color: #e1bee7; margin: 0; font-weight: 500;'>95%+ precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("## 🌟 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(138, 43, 226, 0.08); padding: 1.5rem; 
                    border-radius: 15px; border-left: 5px solid #9d4edd;
                    border: 1px solid rgba(138, 43, 226, 0.2);'>
            <h4 style='color: #c77dff; margin-top: 0; font-weight: 700;'>🔍 Advanced NLP</h4>
            <ul style='color: #e0e0e0; line-height: 1.8;'>
                <li>Text preprocessing & lemmatization</li>
                <li>TF-IDF vectorization</li>
                <li>Stopword removal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(138, 43, 226, 0.08); padding: 1.5rem; 
                    border-radius: 15px; border-left: 5px solid #9d4edd; margin-top: 1rem;
                    border: 1px solid rgba(138, 43, 226, 0.2);'>
            <h4 style='color: #c77dff; margin-top: 0; font-weight: 700;'>📊 Comprehensive Analytics</h4>
            <ul style='color: #e0e0e0; line-height: 1.8;'>
                <li>Model performance comparison</li>
                <li>Word clouds & visualizations</li>
                <li>Statistical insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(138, 43, 226, 0.08); padding: 1.5rem; 
                    border-radius: 15px; border-left: 5px solid #9d4edd;
                    border: 1px solid rgba(138, 43, 226, 0.2);'>
            <h4 style='color: #c77dff; margin-top: 0; font-weight: 700;'>🤖 Multiple Models</h4>
            <ul style='color: #e0e0e0; line-height: 1.8;'>
                <li>Naive Bayes</li>
                <li>Logistic Regression</li>
                <li>Support Vector Machine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(138, 43, 226, 0.08); padding: 1.5rem; 
                    border-radius: 15px; border-left: 5px solid #9d4edd; margin-top: 1rem;
                    border: 1px solid rgba(138, 43, 226, 0.2);'>
            <h4 style='color: #c77dff; margin-top: 0; font-weight: 700;'>⚡ User-Friendly</h4>
            <ul style='color: #e0e0e0; line-height: 1.8;'>
                <li>Intuitive interface</li>
                <li>Real-time predictions</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("## 🚀 Getting Started")
    st.info("""
    **Step 1:** Click **'Train/Reload Models'** in the sidebar to initialize the system  
    **Step 2:** Navigate to **'Predict'** page to analyze news articles  
    **Step 3:** Explore **'Analytics'** for detailed insights and visualizations
    """)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Please click 'Train/Reload Models' in the sidebar.")

elif page == "🔮 Predict":
    st.markdown("# 🔮 Fake News Prediction")
    st.markdown("### *Analyze news articles for authenticity*")
    st.markdown("---")
    
    if not st.session_state.models_trained:
        st.error("❌ Models not trained yet. Please go to sidebar and click 'Train/Reload Models'.")
    else:
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Type/Paste Text", "📄 Sample Articles"],
            horizontal=True
        )
        
        if input_method == "✍️ Type/Paste Text":
            user_input = st.text_area(
                "Enter news article text:",
                height=200,
                placeholder="Paste the news article title and content here..."
            )
            
            if st.button("🔍 Analyze Article", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing..."):
                        # Preprocess
                        processed_text = preprocess_text(user_input)
                        
                        # Get predictions from all models
                        predictions = {}
                        confidences = {}
                        
                        # Naive Bayes
                        X_input = st.session_state.vectorizer_count.transform([processed_text])
                        pred_nb = st.session_state.models['Naive Bayes'].predict(X_input)[0]
                        predictions['Naive Bayes'] = pred_nb
                        
                        # Logistic Regression
                        pred_lr = st.session_state.models['Logistic Regression'].predict(X_input)[0]
                        predictions['Logistic Regression'] = pred_lr
                        
                        # SVM
                        X_input_tfidf = st.session_state.vectorizer_tfidf.transform([processed_text])
                        pred_svm = st.session_state.models['SVM (TF-IDF)'].predict(X_input_tfidf)[0]
                        predictions['SVM (TF-IDF)'] = pred_svm
                        
                        # Ensemble prediction (majority voting)
                        votes = list(predictions.values())
                        final_prediction = 1 if sum(votes) >= 2 else 0
                        confidence = (sum(votes) / len(votes)) * 100 if final_prediction == 1 else ((len(votes) - sum(votes)) / len(votes)) * 100
                        
                        st.markdown("---")
                        
                        # Display result
                        if final_prediction == 1:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #0f5132 0%, #198754 100%); 
                                        padding: 2.5rem; border-radius: 16px; text-align: center;
                                        border: 2px solid #25a867;
                                        box-shadow: 0 8px 30px rgba(37, 168, 103, 0.4);'>
                                <h1 style='color: white; margin: 0; font-size: 3rem; font-weight: 800; 
                                           text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>✅ REAL NEWS</h1>
                                <h3 style='color: #b7e4c7; margin: 1rem 0 0 0; font-weight: 500;'>This article appears to be authentic</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #721c24 0%, #a94442 100%); 
                                        padding: 2.5rem; border-radius: 16px; text-align: center;
                                        border: 2px solid #dc3545;
                                        box-shadow: 0 8px 30px rgba(220, 53, 69, 0.4);'>
                                <h1 style='color: white; margin: 0; font-size: 3rem; font-weight: 800;
                                           text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>⚠️ FAKE NEWS</h1>
                                <h3 style='color: #f8d7da; margin: 1rem 0 0 0; font-weight: 500;'>This article may contain false information</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Confidence", f"{confidence:.1f}%")
                        col2.metric("Models Agreeing", f"{max(sum(votes), len(votes)-sum(votes))}/3")
                        col3.metric("Word Count", len(processed_text.split()))
                        
                        # Model predictions breakdown
                        with st.expander("📊 Model Predictions Breakdown"):
                            pred_df = pd.DataFrame([
                                {'Model': model, 
                                 'Prediction': '✅ Real' if pred == 1 else '⚠️ Fake',
                                 'Label': pred}
                                for model, pred in predictions.items()
                            ])
                            
                            fig = px.bar(
                                pred_df, x='Model', y='Label',
                                color='Prediction',
                                color_discrete_map={'✅ Real': '#4caf50', '⚠️ Fake': '#f44336'},
                                title='Individual Model Predictions'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#e0e0e0'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.warning("⚠️ Please enter some text to analyze.")
        
        else:  # Sample Articles
            st.markdown("### Select a sample article to analyze:")
            
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Sample Fake News")
                fake_samples = df[df['label'] == 0].sample(3, random_state=42)
                for idx, (i, row) in enumerate(fake_samples.iterrows()):
                    if st.button(f"Fake Sample {idx+1}", key=f"fake_{i}"):
                        st.session_state.selected_sample = row['text']
                        st.session_state.selected_label = 0
            
            with col2:
                st.markdown("#### 🟢 Sample Real News")
                real_samples = df[df['label'] == 1].sample(3, random_state=42)
                for idx, (i, row) in enumerate(real_samples.iterrows()):
                    if st.button(f"Real Sample {idx+1}", key=f"real_{i}"):
                        st.session_state.selected_sample = row['text']
                        st.session_state.selected_label = 1
            
            if 'selected_sample' in st.session_state:
                st.markdown("---")
                st.markdown("#### Selected Article:")
                st.text_area("", st.session_state.selected_sample[:500] + "...", height=150, disabled=True)
                
                if st.button("🔍 Analyze This Article", type="primary"):
                    with st.spinner("Analyzing..."):
                        processed_text = st.session_state.selected_sample
                        
                        # Get predictions
                        predictions = {}
                        X_input = st.session_state.vectorizer_count.transform([processed_text])
                        predictions['Naive Bayes'] = st.session_state.models['Naive Bayes'].predict(X_input)[0]
                        predictions['Logistic Regression'] = st.session_state.models['Logistic Regression'].predict(X_input)[0]
                        
                        X_input_tfidf = st.session_state.vectorizer_tfidf.transform([processed_text])
                        predictions['SVM (TF-IDF)'] = st.session_state.models['SVM (TF-IDF)'].predict(X_input_tfidf)[0]
                        
                        votes = list(predictions.values())
                        final_prediction = 1 if sum(votes) >= 2 else 0
                        confidence = (sum(votes) / len(votes)) * 100 if final_prediction == 1 else ((len(votes) - sum(votes)) / len(votes)) * 100
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Prediction:")
                            if final_prediction == 1:
                                st.success("✅ REAL NEWS")
                            else:
                                st.error("⚠️ FAKE NEWS")
                        
                        with col2:
                            st.markdown("### Actual:")
                            if st.session_state.selected_label == 1:
                                st.success("✅ REAL NEWS")
                            else:
                                st.error("⚠️ FAKE NEWS")
                        
                        if final_prediction == st.session_state.selected_label:
                            st.success("🎯 Correct Prediction!")
                        else:
                            st.warning("❌ Incorrect Prediction")
                        
                        st.metric("Confidence", f"{confidence:.1f}%")

elif page == "📊 Analytics":
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("### *Comprehensive model insights and visualizations*")
    st.markdown("---")
    
    if not st.session_state.models_trained:
        st.error("❌ Models not trained yet. Please go to sidebar and click 'Train/Reload Models'.")
    else:
        tabs = st.tabs(["📈 Model Performance", "☁️ Word Clouds", "📊 Data Analysis"])
        
        # Tab 1: Model Performance
        with tabs[0]:
            st.markdown("## 🏆 Model Comparison")
            
            results = st.session_state.results
            
            # Metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            best_acc = results.loc[results['Accuracy'].idxmax()]
            best_f1 = results.loc[results['F1-Score'].idxmax()]
            best_prec = results.loc[results['Precision'].idxmax()]
            best_rec = results.loc[results['Recall'].idxmax()]
            
            with col1:
                st.metric("Best Accuracy", f"{best_acc['Accuracy']:.4f}", best_acc['Model'])
            with col2:
                st.metric("Best F1-Score", f"{best_f1['F1-Score']:.4f}", best_f1['Model'])
            with col3:
                st.metric("Best Precision", f"{best_prec['Precision']:.4f}", best_prec['Model'])
            with col4:
                st.metric("Best Recall", f"{best_rec['Recall']:.4f}", best_rec['Model'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Performance table
            st.markdown("### 📋 Detailed Performance Metrics")
            
            # Style the dataframe
            styled_results = results.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'ROC-AUC': '{:.4f}'
            }).background_gradient(cmap='Blues', subset=['Accuracy', 'F1-Score', 'ROC-AUC'])
            
            st.dataframe(styled_results, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart comparison
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=results['Model'],
                        y=results[metric],
                        text=results[metric].round(3),
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    barmode='group',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0',
                    xaxis_title='Model',
                    yaxis_title='Score'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Radar chart with purple theme
                fig = go.Figure()
                
                # Purple/Lavender color palette matching theme
                purple_colors = ['#9d4edd', '#c77dff', '#e0aaff']
                
                for idx, row in results.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
                        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        fill='toself',
                        name=row['Model'],
                        line_color=purple_colors[idx % len(purple_colors)],
                        fillcolor=purple_colors[idx % len(purple_colors)],
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, 
                            range=[0, 1],
                            gridcolor='rgba(138, 43, 226, 0.2)'
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(138, 43, 226, 0.2)'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=True,
                    title='Model Performance Radar',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Word Clouds
        with tabs[1]:
            st.markdown("## ☁️ Word Cloud Analysis")
            
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔴 Fake News Word Cloud")
                fake_text = ' '.join(df[df['label'] == 0]['text'])
                wordcloud_fake = WordCloud(
                    width=800, height=400,
                    background_color='#000000',
                    colormap='Reds',
                    max_words=100
                ).generate(fake_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_fake, interpolation='bilinear')
                ax.axis('off')
                ax.set_facecolor('#000000')
                fig.patch.set_facecolor('#000000')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 🟢 Real News Word Cloud")
                real_text = ' '.join(df[df['label'] == 1]['text'])
                wordcloud_real = WordCloud(
                    width=800, height=400,
                    background_color='#000000',
                    colormap='Greens',
                    max_words=100
                ).generate(real_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_real, interpolation='bilinear')
                ax.axis('off')
                ax.set_facecolor('#000000')
                fig.patch.set_facecolor('#000000')
                st.pyplot(fig)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top words
            st.markdown("## 📝 Most Frequent Words")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Top 20 Words in Fake News")
                fake_words = ' '.join(df[df['label'] == 0]['text']).split()
                fake_counter = Counter(fake_words)
                top_fake = pd.DataFrame(fake_counter.most_common(20), columns=['Word', 'Count'])
                
                fig = px.bar(
                    top_fake, x='Count', y='Word',
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0',
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Top 20 Words in Real News")
                real_words = ' '.join(df[df['label'] == 1]['text']).split()
                real_counter = Counter(real_words)
                top_real = pd.DataFrame(real_counter.most_common(20), columns=['Word', 'Count'])
                
                fig = px.bar(
                    top_real, x='Count', y='Word',
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0',
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Data Analysis
        with tabs[2]:
            st.markdown("## 📊 Dataset Statistics")
            
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Articles", f"{len(df):,}")
            with col2:
                st.metric("Fake Articles", f"{(df['label']==0).sum():,}")
            with col3:
                st.metric("Real Articles", f"{(df['label']==1).sum():,}")
            with col4:
                st.metric("Avg Words/Article", f"{df['word_count'].mean():.0f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Article length distribution
            st.markdown("### 📏 Article Length Distribution")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[df['label'] == 0]['word_count'],
                name='Fake News',
                opacity=0.7,
                marker_color='#f44336',
                nbinsx=50
            ))
            
            fig.add_trace(go.Histogram(
                x=df[df['label'] == 1]['word_count'],
                name='Real News',
                opacity=0.7,
                marker_color='#4caf50',
                nbinsx=50
            ))
            
            fig.update_layout(
                barmode='overlay',
                title='Word Count Distribution: Fake vs Real News',
                xaxis_title='Number of Words',
                yaxis_title='Frequency',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Category distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥧 Label Distribution")
                labels = ['Fake News', 'Real News']
                values = [(df['label']==0).sum(), (df['label']==1).sum()]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.4,
                    marker_colors=['#f44336', '#4caf50']
                )])
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Statistical Summary")
                summary_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Fake News': [
                        df[df['label']==0]['word_count'].mean(),
                        df[df['label']==0]['word_count'].median(),
                        df[df['label']==0]['word_count'].std(),
                        df[df['label']==0]['word_count'].min(),
                        df[df['label']==0]['word_count'].max()
                    ],
                    'Real News': [
                        df[df['label']==1]['word_count'].mean(),
                        df[df['label']==1]['word_count'].median(),
                        df[df['label']==1]['word_count'].std(),
                        df[df['label']==1]['word_count'].min(),
                        df[df['label']==1]['word_count'].max()
                    ]
                })
                
                summary_df['Fake News'] = summary_df['Fake News'].round(2)
                summary_df['Real News'] = summary_df['Real News'].round(2)
                
                st.dataframe(summary_df, use_container_width=True)

else:  # About page
    st.markdown("# About")
    st.markdown("### *Learn more about this application*")
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This **Fake News Detection System** is an advanced machine learning application designed to 
    identify and classify news articles as either authentic or potentially misleading. Built using 
    state-of-the-art NLP techniques and multiple classification algorithms, the system provides 
    robust predictions with high accuracy.
    
    ## 🔬 Technical Details
    
    ### Machine Learning Models:
    - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
    - **Logistic Regression**: Linear model for binary classification
    - **Support Vector Machine**: Maximum-margin classifier with linear kernel
    
    ### NLP Pipeline:
    1. **Text Cleaning**: Remove special characters, numbers, and extra whitespace
    2. **Lowercasing**: Normalize text to lowercase
    3. **Stopword Removal**: Remove common words that don't add semantic value
    4. **Lemmatization**: Reduce words to their base/dictionary form
    5. **Vectorization**: Convert text to numerical features using TF-IDF and Count Vectorizer
    
    ### Dataset:
    - Source: Fake and Real News Dataset
    - Total Articles: 44,000+
    - Classes: Binary (Fake/Real)
    - Features: Title and Article Text
    
    ## 📊 Performance Metrics
    
    The models are evaluated using:
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive predictions that are correct
    - **Recall**: Proportion of actual positives correctly identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Model's ability to distinguish between classes
    
    ## 🛠️ Technologies Used
    
    - **Frontend**: Streamlit
    - **ML Libraries**: Scikit-learn
    - **NLP**: NLTK
    - **Visualization**: Plotly, Matplotlib, WordCloud
    - **Data Processing**: Pandas, NumPy
    
    ## 👨‍💻 Developer
    
    Created as part of NLP Internship Projects
    
    ## 📝 Note
    
    This system is designed for educational and research purposes. While it achieves high accuracy,
    no automated system is perfect. Always verify important information from multiple reliable sources.
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #c77dff; font-size: 1rem;'>
        <p>© 2026 Fake News Detection System | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #c77dff; padding: 1rem;'>
    <p>💡 Tip: Train the models first from the sidebar, then explore predictions and analytics!</p>
</div>
""", unsafe_allow_html=True)
