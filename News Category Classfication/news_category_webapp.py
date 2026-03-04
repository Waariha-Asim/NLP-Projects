import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

# Machine Learning
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Download NLTK resources
@st.cache_resource
def download_nltk():
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk()

# Page Configuration
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Modern Black Theme with Glassmorphism & Blue Accents
st.markdown("""
<style>
    /* Main background - Pure Black with subtle pattern */
    .stApp {
        background: #000000;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(13, 71, 161, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(25, 118, 210, 0.06) 0%, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Sidebar - Glassmorphism with blue accent */
    [data-testid="stSidebar"] {
        background: rgba(5, 10, 20, 0.85);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(25, 118, 210, 0.3);
        box-shadow: 4px 0 24px rgba(13, 71, 161, 0.15);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #64b5f6 !important;
        text-shadow: 0 0 20px rgba(100, 181, 246, 0.4);
    }
    
    /* Headers - Elegant Blue with Glow */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-align: center;
        letter-spacing: -1px !important;
        text-shadow: 0 0 40px rgba(66, 165, 245, 0.6);
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #42a5f5, #64b5f6, #42a5f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #90caf9 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 0 15px rgba(144, 202, 249, 0.3);
    }
    
    h3 {
        color: #bbdefb !important;
        font-weight: 600 !important;
    }
    
    /* Buttons - Blue Glassmorphism */
    .stButton>button {
        background: rgba(25, 118, 210, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #64b5f6;
        border: 2px solid rgba(100, 181, 246, 0.4);
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        transition: all 0.4s ease;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background: rgba(25, 118, 210, 0.4);
        border-color: #42a5f5;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(66, 165, 245, 0.5);
        transform: translateY(-3px) scale(1.02);
    }
    
    /* Text Input - Glassmorphism */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div {
        background: rgba(13, 27, 42, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #e3f2fd;
        border: 1px solid rgba(66, 165, 245, 0.3);
        border-radius: 10px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stTextInput>div>div>input:focus, 
    .stTextArea>div>div>textarea:focus {
        border-color: #42a5f5;
        box-shadow: 0 0 20px rgba(66, 165, 245, 0.4);
        background: rgba(13, 27, 42, 0.8);
    }
    
    /* Metrics - Glowing Blue */
    [data-testid="stMetricValue"] {
        color: #42a5f5 !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 25px rgba(66, 165, 245, 0.7);
    }
    
    [data-testid="stMetricLabel"] {
        color: #90caf9 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Info/Success/Warning boxes - Glassmorphism */
    .stAlert {
        background: rgba(13, 71, 161, 0.2);
        backdrop-filter: blur(10px);
        border-left: 4px solid #42a5f5;
        border-radius: 12px;
        color: #bbdefb;
        box-shadow: 0 4px 16px rgba(13, 71, 161, 0.2);
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #90caf9 !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Dataframe - Glassmorphism */
    .dataframe {
        background: rgba(10, 25, 41, 0.7) !important;
        backdrop-filter: blur(10px);
        color: #e3f2fd !important;
        border: 1px solid rgba(66, 165, 245, 0.3) !important;
        border-radius: 10px;
    }
    
    /* Expander - Glassmorphism */
    .streamlit-expanderHeader {
        background: rgba(13, 71, 161, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        color: #64b5f6 !important;
        font-weight: 700;
        border: 1px solid rgba(66, 165, 245, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(13, 71, 161, 0.25);
        border-color: #42a5f5;
        box-shadow: 0 4px 16px rgba(66, 165, 245, 0.2);
    }
    
    /* Success/Error boxes - Glassmorphism */
    .success-box {
        background: rgba(46, 125, 50, 0.2);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #4caf50;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.2);
    }
    
    .category-box {
        background: rgba(13, 27, 42, 0.6);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(66, 165, 245, 0.3);
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.2);
        transition: all 0.3s ease;
    }
    
    .category-box:hover {
        border-color: #42a5f5;
        box-shadow: 0 12px 40px rgba(66, 165, 245, 0.4);
        transform: translateY(-5px);
    }
    
    /* Scrollbar - Blue Theme */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.4);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1976d2 0%, #42a5f5 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #42a5f5 0%, #64b5f6 100%);
    }
    
    /* Cards - Glassmorphism */
    .card {
        background: rgba(13, 27, 42, 0.5);
        backdrop-filter: blur(15px) saturate(180%);
        -webkit-backdrop-filter: blur(15px) saturate(180%);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(66, 165, 245, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(66, 165, 245, 0.5);
        box-shadow: 0 12px 40px rgba(66, 165, 245, 0.3);
        transform: translateY(-5px);
    }
    
    /* Links - Blue Glow */
    a {
        color: #64b5f6 !important;
        text-decoration: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    a:hover {
        color: #90caf9 !important;
        text-shadow: 0 0 15px rgba(100, 181, 246, 0.6);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #90caf9 !important;
        font-weight: 700;
        font-size: 1.05rem;
    }
    
    /* Tabs - Glassmorphism */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(13, 27, 42, 0.4);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #90caf9;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(25, 118, 210, 0.3);
        backdrop-filter: blur(10px);
        color: #ffffff;
        border: 1px solid rgba(66, 165, 245, 0.5);
        box-shadow: 0 4px 16px rgba(66, 165, 245, 0.3);
    }
    
    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #64b5f6 !important;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #90caf9 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Category mapping
CATEGORY_MAP = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

CATEGORY_COLORS = {
    'World': '#1976d2',
    'Sports': '#42a5f5',
    'Business': '#64b5f6',
    'Sci/Tech': '#90caf9'
}

CATEGORY_EMOJIS = {
    'World': '🌍',
    'Sports': '⚽',
    'Business': '💼',
    'Sci/Tech': '🔬'
}

# Preprocessing function
@st.cache_data
def preprocess_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Load data
@st.cache_data
def load_and_preprocess_data(sample_size=10000):
    """Load and preprocess AG News dataset"""
    try:
        with st.spinner('📊 Loading AG News dataset...'):
            train_df = pd.read_csv(r'C:\Users\AR FAST\Documents\NLP Internship Projects\News Category Classification\train.csv')
            test_df = pd.read_csv(r'C:\Users\AR FAST\Documents\NLP Internship Projects\News Category Classification\test.csv')
            
            # Rename columns
            train_df.columns = ['Class Index', 'Title', 'Description']
            test_df.columns = ['Class Index', 'Title', 'Description']
            
            # Map categories
            train_df['Category'] = train_df['Class Index'].map(CATEGORY_MAP)
            test_df['Category'] = test_df['Class Index'].map(CATEGORY_MAP)
            
            # Combine title and description
            train_df['Text'] = train_df['Title'] + ' ' + train_df['Description']
            test_df['Text'] = test_df['Title'] + ' ' + test_df['Description']
            
            # Sample
            train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
            test_sample = test_df.sample(n=min(sample_size//2, len(test_df)), random_state=42)
        
        with st.spinner('🔄 Preprocessing text...'):
            train_sample['Processed_Text'] = train_sample['Text'].apply(preprocess_text)
            test_sample['Processed_Text'] = test_sample['Text'].apply(preprocess_text)
        
        return train_sample, test_sample
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please download AG News from Kaggle.")
        st.info("📥 **Dataset Link:** https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")
        return None, None

# Train models
@st.cache_resource
def train_models(train_df, test_df):
    """Train all classification models"""
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['Processed_Text'])
    X_test = vectorizer.transform(test_df['Processed_Text'])
    
    y_train = train_df['Class Index'].values - 1
    y_test = test_df['Class Index'].values - 1
    
    models = {}
    results = []
    predictions = {}
    
    # Logistic Regression
    with st.spinner('🤖 Training Logistic Regression...'):
        lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        models['Logistic Regression'] = lr_model
        predictions['Logistic Regression'] = lr_pred
        
        results.append({
            'Model': 'Logistic Regression',
            'Accuracy': accuracy_score(y_test, lr_pred),
            'Precision': precision_score(y_test, lr_pred, average='weighted'),
            'Recall': recall_score(y_test, lr_pred, average='weighted'),
            'F1-Score': f1_score(y_test, lr_pred, average='weighted')
        })
    
    # Random Forest
    with st.spinner('🤖 Training Random Forest...'):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        models['Random Forest'] = rf_model
        predictions['Random Forest'] = rf_pred
        
        results.append({
            'Model': 'Random Forest',
            'Accuracy': accuracy_score(y_test, rf_pred),
            'Precision': precision_score(y_test, rf_pred, average='weighted'),
            'Recall': recall_score(y_test, rf_pred, average='weighted'),
            'F1-Score': f1_score(y_test, rf_pred, average='weighted')
        })
    
    # SVM
    with st.spinner('🤖 Training SVM...'):
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        models['SVM'] = svm_model
        predictions['SVM'] = svm_pred
        
        results.append({
            'Model': 'SVM',
            'Accuracy': accuracy_score(y_test, svm_pred),
            'Precision': precision_score(y_test, svm_pred, average='weighted'),
            'Recall': recall_score(y_test, svm_pred, average='weighted'),
            'F1-Score': f1_score(y_test, svm_pred, average='weighted')
        })
    
    # XGBoost
    with st.spinner('🤖 Training XGBoost...'):
        xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        models['XGBoost'] = xgb_model
        predictions['XGBoost'] = xgb_pred
        
        results.append({
            'Model': 'XGBoost',
            'Accuracy': accuracy_score(y_test, xgb_pred),
            'Precision': precision_score(y_test, xgb_pred, average='weighted'),
            'Recall': recall_score(y_test, xgb_pred, average='weighted'),
            'F1-Score': f1_score(y_test, xgb_pred, average='weighted')
        })
    
    return models, pd.DataFrame(results), vectorizer, y_test, predictions

# Sidebar
with st.sidebar:
    st.markdown("# 📰 News Classifier")
    st.markdown("### *AG News Dataset*")
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration")
    
    sample_size = st.selectbox(
        "Training Sample Size",
        options=[5000, 10000, 15000, 20000],
        index=1,
        help="Number of samples for training"
    )
    
    if st.button("🚀 TRAIN MODELS", use_container_width=True):
        with st.spinner("Training in progress..."):
            train_df, test_df = load_and_preprocess_data(sample_size=sample_size)
            if train_df is not None:
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
                models, results, vectorizer, y_test, predictions = train_models(train_df, test_df)
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.vectorizer = vectorizer
                st.session_state.y_test = y_test
                st.session_state.predictions = predictions
                st.session_state.models_trained = True
                st.success("✅ Training Complete!")
                st.rerun()
    
    st.markdown("---")
    
    if st.session_state.models_trained:
        st.markdown("### 📊 Quick Stats")
        st.metric("Train Size", f"{len(st.session_state.train_df):,}")
        st.metric("Test Size", f"{len(st.session_state.test_df):,}")
        st.metric("Categories", "4")
        
        best_model = st.session_state.results.loc[st.session_state.results['Accuracy'].idxmax()]
        st.markdown(f"### 🏆 Best Model")
        st.info(f"**{best_model['Model']}**\n\nAccuracy: {best_model['Accuracy']:.4f}")
    
    st.markdown("---")
    st.markdown("###  Categories")
    for cat, emoji in CATEGORY_EMOJIS.items():
        st.markdown(f"{emoji} **{cat}**")

# Header
st.markdown("# 📰 News Category Classification Dashboard")
st.markdown("### *Multiclass Text Classification with Machine Learning*")
st.markdown("---")

if not st.session_state.models_trained:
    st.markdown("""
    <div class='category-box' style='text-align: center; padding: 3rem;'>
        <h2 style='color: #64b5f6; margin: 0;'>📰 Welcome to News Category Classifier</h2>
        <p style='color: #90caf9; font-size: 1.3rem; margin: 2rem 0;'>
            Classify news articles into 4 categories using AI
        </p>
        <p style='color: #90caf9; font-size: 1.1rem; margin: 0;'>
            🌍 <strong>World</strong> | ⚽ <strong>Sports</strong> | 💼 <strong>Business</strong> | 🔬 <strong>Sci/Tech</strong>
        </p>
        <p style='color: #42a5f5; font-size: 1.2rem; margin: 2rem 0 0 0; font-weight: 700;'>
            👈 Click <strong>'TRAIN MODELS'</strong> in the sidebar to begin
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>🤖</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>4 ML Models</h3>
            <p style='color: #90caf9; margin: 0;'>Advanced Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>📊</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>4 Categories</h3>
            <p style='color: #90caf9; margin: 0;'>Multiclass Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>☁️</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>Word Clouds</h3>
            <p style='color: #90caf9; margin: 0;'>Per Category Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>⚡</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>Real-time</h3>
            <p style='color: #90caf9; margin: 0;'>Instant Predictions</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "🔮 Predict", "📊 Performance", "☁️ Word Clouds", "ℹ️ About"])
    
    # Tab 1: Dashboard
    with tab1:
        st.markdown("## 📊 Overview & Statistics")
        
        train_df = st.session_state.train_df
        results = st.session_state.results
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Total Articles", f"{len(train_df):,}")
        col2.metric("World 🌍", f"{(train_df['Category']=='World').sum():,}")
        col3.metric("Sports ⚽", f"{(train_df['Category']=='Sports').sum():,}")
        col4.metric("Business 💼", f"{(train_df['Category']=='Business').sum():,}")
        col5.metric("Sci/Tech 🔬", f"{(train_df['Category']=='Sci/Tech').sum():,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Best model showcase
        best_model = results.loc[results['Accuracy'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='category-box' style='text-align: center;'>
                <h3 style='color: #64b5f6;'>🏆 Best Model</h3>
                <h2 style='color: white; margin: 1rem 0;'>{best_model['Model']}</h2>
                <p style='color: #90caf9; font-size: 1.8rem; margin: 0; font-weight: 700;'>{best_model['Accuracy']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='category-box' style='text-align: center;'>
                <h3 style='color: #64b5f6;'>📈 Avg F1-Score</h3>
                <h2 style='color: white; margin: 1rem 0;'>All Models</h2>
                <p style='color: #90caf9; font-size: 1.8rem; margin: 0; font-weight: 700;'>{results['F1-Score'].mean():.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='category-box' style='text-align: center;'>
                <h3 style='color: #64b5f6;'>🎯 Avg Precision</h3>
                <h2 style='color: white; margin: 1rem 0;'>All Models</h2>
                <p style='color: #90caf9; font-size: 1.8rem; margin: 0; font-weight: 700;'>{results['Precision'].mean():.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Category Distribution")
            category_counts = train_df['Category'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=[f"{CATEGORY_EMOJIS[cat]} {cat}" for cat in category_counts.index],
                values=category_counts.values,
                hole=0.4,
                marker_colors=[CATEGORY_COLORS[cat] for cat in category_counts.index]
            )])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#90caf9'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Model Accuracy Comparison")
            
            fig = go.Figure(data=[go.Bar(
                x=results['Model'],
                y=results['Accuracy'],
                text=results['Accuracy'].round(4),
                textposition='auto',
                marker_color='#42a5f5',
                marker_line_color='#1976d2',
                marker_line_width=2
            )])
            
            fig.update_layout(
                xaxis_title='Model',
                yaxis_title='Accuracy',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#90caf9',
                yaxis_range=[0, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.markdown("### 📋 Detailed Performance Metrics")
        
        styled_results = results.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}'
        }).background_gradient(cmap='YlOrRd', subset=['Accuracy', 'F1-Score'])
        
        st.dataframe(styled_results, use_container_width=True)
    
    # Tab 2: Predict
    with tab2:
        st.markdown("## 🔮 News Article Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter news article (title and description):",
                height=200,
                placeholder="Type or paste a news article here...\n\nExample: 'Olympic Games to be held next summer. Athletes from around the world will compete...'"
            )
        
        with col2:
            st.markdown("### 📚 Choose Model")
            model_choice = st.radio(
                "",
                options=['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost'],
                index=0
            )
        
        if st.button("🔍 CLASSIFY ARTICLE", use_container_width=True, type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing article..."):
                    processed = preprocess_text(user_input)
                    X_input = st.session_state.vectorizer.transform([processed])
                    prediction = st.session_state.models[model_choice].predict(X_input)[0]
                    
                    category = CATEGORY_MAP[prediction + 1]
                    emoji = CATEGORY_EMOJIS[category]
                    color = CATEGORY_COLORS[category]
                    
                    st.markdown("---")
                    
                    st.markdown(f"""
                    <div class='category-box' style='text-align: center; border-color: {color};'>
                        <h1 style='color: {color}; margin: 0; font-size: 4rem;'>{emoji}</h1>
                        <h1 style='color: white; margin: 1rem 0;'>{category.upper()}</h1>
                        <p style='color: #90caf9; font-size: 1.2rem; margin: 0;'>Predicted Category</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Get predictions from all models
                    all_predictions = {}
                    for model_name, model in st.session_state.models.items():
                        X_input = st.session_state.vectorizer.transform([processed])
                        pred = model.predict(X_input)[0]
                        all_predictions[model_name] = CATEGORY_MAP[pred + 1]
                    
                    with st.expander("📊 See All Model Predictions"):
                        pred_df = pd.DataFrame([
                            {'Model': model, 'Prediction': f"{CATEGORY_EMOJIS[cat]} {cat}"}
                            for model, cat in all_predictions.items()
                        ])
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Please enter a news article to classify.")
        
        st.markdown("---")
        
        # Sample articles
        st.markdown("### 📄 Or Try Sample Articles")
        
        samples = {
            'World': "United Nations Security Council meets to discuss international peace efforts. Diplomatic relations between nations...",
            'Sports': "Olympic champion breaks world record in 100m sprint. The athlete dominated the competition...",
            'Business': "Stock market reaches all-time high as tech companies report strong quarterly earnings. Investors celebrate...",
            'Sci/Tech': "Scientists discover new exoplanet with potential for life. The research team used advanced telescopes..."
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        cols = [col1, col2, col3, col4]
        for idx, (cat, sample) in enumerate(samples.items()):
            with cols[idx]:
                if st.button(f"{CATEGORY_EMOJIS[cat]} {cat}", key=f"sample_{cat}", use_container_width=True):
                    st.session_state.sample_text = sample
                    st.rerun()
        
        if 'sample_text' in st.session_state:
            st.text_area("Selected Sample:", st.session_state.sample_text, height=100, disabled=True)
    
    # Tab 3: Performance
    with tab3:
        st.markdown("## 📊 Model Performance Analysis")
        
        results = st.session_state.results
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Metrics Comparison")
            
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['#1976d2', '#42a5f5', '#64b5f6', '#90caf9']
            
            for idx, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results['Model'],
                    y=results[metric],
                    text=results[metric].round(3),
                    textposition='auto',
                    marker_color=colors[idx]
                ))
            
            fig.update_layout(
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#90caf9',
                xaxis_title='Model',
                yaxis_title='Score',
                legend=dict(orientation='h', y=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Radar Chart")
            
            fig = go.Figure()
            
            colors = ['#1976d2', '#42a5f5', '#64b5f6', '#90caf9']
            
            for idx, row in results.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=row['Model'],
                    line_color=colors[idx],
                    fillcolor=colors[idx],
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor='rgba(255, 143, 0, 0.3)'
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255, 143, 0, 0.3)'
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#90caf9',
                legend=dict(orientation='h', y=-0.2)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("### 🔥 Confusion Matrix - Logistic Regression")
        
        lr_pred = st.session_state.predictions['Logistic Regression']
        cm = confusion_matrix(st.session_state.y_test, lr_pred)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['World', 'Sports', 'Business', 'Sci/Tech'],
            y=['World', 'Sports', 'Business', 'Sci/Tech'],
            color_continuous_scale='YlOrRd',
            text_auto=True
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#90caf9'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Word Clouds
    with tab4:
        st.markdown("## ☁️ Word Cloud Analysis by Category")
        
        train_df = st.session_state.train_df
        
        # Word clouds
        col1, col2 = st.columns(2)
        
        categories = sorted(train_df['Category'].unique())
        colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
        
        for idx, category in enumerate(categories):
            with (col1 if idx % 2 == 0 else col2):
                st.markdown(f"### {CATEGORY_EMOJIS[category]} {category}")
                
                category_text = ' '.join(train_df[train_df['Category'] == category]['Processed_Text'])
                
                wordcloud = WordCloud(
                    width=800, height=500,
                    background_color='#1a1a1a',
                    colormap=colormaps[idx],
                    max_words=100,
                    relative_scaling=0.5
                ).generate(category_text)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#1a1a1a')
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Top words per category
        st.markdown("## 📊 Top 15 Words per Category")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, category in enumerate(categories):
            category_text = ' '.join(train_df[train_df['Category'] == category]['Processed_Text'])
            words = category_text.split()
            word_freq = Counter(words).most_common(15)
            
            words_list = [w[0] for w in word_freq]
            freq_list = [w[1] for w in word_freq]
            
            axes[idx].barh(words_list, freq_list, color=['#1976d2', '#42a5f5', '#64b5f6', '#90caf9'][idx])
            axes[idx].set_xlabel('Frequency', fontsize=11, fontweight='bold', color='#90caf9')
            axes[idx].set_title(f'{CATEGORY_EMOJIS[category]} {category}', fontsize=13, fontweight='bold', color='#90caf9')
            axes[idx].invert_yaxis()
            axes[idx].tick_params(colors='#90caf9')
            axes[idx].set_facecolor('#1a1a1a')
            axes[idx].grid(axis='x', alpha=0.3, color='#42a5f5')
        
        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Tab 5: About
    with tab5:
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        This **News Category Classification** system classifies news articles into 4 categories:
        - 🌍 **World**: International news, politics, diplomacy
        - ⚽ **Sports**: Athletics, competitions, sports events
        - 💼 **Business**: Finance, economy, markets, business news
        - 🔬 **Sci/Tech**: Science, technology, research, innovation
        
        ## 📊 Dataset
        
        **AG News Classification Dataset** from Kaggle
        - 📥 [Download Link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
        - Total Articles: 120,000 (Train) + 7,600 (Test)
        - Balanced across 4 categories
        - Contains title and description for each article
        
        ## 🤖 Machine Learning Models
        
        ### Traditional ML:
        1. **Logistic Regression**: Multinomial classifier for multiclass problems
        2. **Random Forest**: Ensemble of decision trees
        3. **Support Vector Machine (SVM)**: Linear kernel classifier
        4. **XGBoost**: Gradient boosting for high performance (Bonus)
        
        ## 🔬 NLP Pipeline
        
        1. **Text Preprocessing**:
           - Lowercase conversion
           - Special character removal
           - Tokenization
           - Stopword removal
           - Lemmatization
        
        2. **Feature Engineering**:
           - TF-IDF Vectorization
           - N-grams (1,2)
           - Max 5000 features
        
        3. **Model Training**:
           - Multiclass classification
           - Weighted metrics for balanced evaluation
        
        ## 📈 Performance Metrics
        
        - **Accuracy**: Overall classification correctness
        - **Precision**: Correct positive predictions per class
        - **Recall**: Coverage of actual positives per class
        - **F1-Score**: Harmonic mean of precision and recall
        
        ## 🛠️ Technologies
        
        - **Framework**: Streamlit
        - **ML**: Scikit-learn, XGBoost
        - **NLP**: NLTK
        - **Visualization**: Plotly, Matplotlib, WordCloud, Seaborn
        - **Data**: Pandas, NumPy
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64b5f6; padding: 1rem;'>
    <p style='font-size: 1.1rem;'>© 2026 News Category Classification | Powered by Machine Learning & Deep Learning</p>
</div>
""", unsafe_allow_html=True)
