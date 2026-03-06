import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# Download NLTK resources
@st.cache_resource
def download_nltk():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk()

# Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Dark Blue Modern Theme (No Sidebar)
st.markdown("""
<style>
    /* Main background - Deep Navy Blue */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2744 50%, #0a1628 100%);
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Headers - Elegant White & Light Blue */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-align: center;
        letter-spacing: -1px !important;
        text-shadow: 0 0 30px rgba(100, 181, 246, 0.4);
        padding: 1rem 0;
    }
    
    h2 {
        color: #e3f2fd !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    h3 {
        color: #bbdefb !important;
        font-weight: 600 !important;
    }
    
    /* Buttons - Navy Blue Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        transition: all 0.4s ease;
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.4);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        box-shadow: 0 8px 30px rgba(21, 101, 192, 0.6);
        transform: translateY(-3px);
    }
    
    /* Text Input - Dark Theme */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div {
        background-color: #0d1b2a;
        color: #e3f2fd;
        border: 2px solid #1565c0;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, 
    .stTextArea>div>div>textarea:focus {
        border-color: #42a5f5;
        box-shadow: 0 0 20px rgba(66, 165, 245, 0.3);
    }
    
    /* Metrics - Eye-catching */
    [data-testid="stMetricValue"] {
        color: #64b5f6 !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 15px rgba(100, 181, 246, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #90caf9 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
    }
    
    /* Tabs - Modern Top Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(90deg, #0d1b2a 0%, #1a2744 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(21, 101, 192, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #90caf9;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(21, 101, 192, 0.2);
        color: #64b5f6;
        border-color: rgba(21, 101, 192, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.5);
        border-color: #42a5f5;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(13, 71, 161, 0.15);
        border-left: 5px solid #1976d2;
        border-radius: 12px;
        color: #e3f2fd;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #90caf9 !important;
        font-weight: 500;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #0d1b2a !important;
        color: #e3f2fd !important;
        border: 1px solid #1565c0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(21, 101, 192, 0.15) 0%, rgba(25, 118, 210, 0.15) 100%);
        border-radius: 10px;
        color: #90caf9 !important;
        font-weight: 700;
        border: 1px solid rgba(21, 101, 192, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(21, 101, 192, 0.25) 0%, rgba(25, 118, 210, 0.25) 100%);
        border-color: #1976d2;
    }
    
    /* Success/Error boxes */
    .success-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #4caf50;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #f44336;
        box-shadow: 0 6px 20px rgba(244, 67, 54, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1628;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1565c0 0%, #1976d2 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1976d2 0%, #42a5f5 100%);
    }
    
    /* Cards */
    .card {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.8) 0%, rgba(26, 39, 68, 0.8) 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 2px solid rgba(21, 101, 192, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: #1976d2;
        box-shadow: 0 12px 35px rgba(21, 101, 192, 0.4);
        transform: translateY(-5px);
    }
    
    /* Links */
    a {
        color: #64b5f6 !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #90caf9 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1565c0 0%, #42a5f5 100%);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #90caf9 !important;
        font-weight: 600;
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
    text = re.sub('<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Load data
@st.cache_data
def load_and_preprocess_data(sample_size=10000):
    """Load and preprocess IMDB dataset"""
    try:
        with st.spinner('📊 Loading IMDB dataset...'):
            df = pd.read_csv(r'C:\Users\AR FAST\Documents\NLP Internship Projects\Sentiment Analysis of Movie Reviews\IMDB Dataset.csv')
            
            # Encode labels
            le = LabelEncoder()
            df['sentiment'] = le.fit_transform(df['sentiment'])
            
            # Sample for faster training
            df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        
        with st.spinner('🔄 Preprocessing reviews...'):
            df['review'] = df['review'].astype(str).apply(preprocess_text)
            df['review_length'] = df['review'].apply(lambda x: len(x.split()))
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Train models
@st.cache_resource
def train_models(df):
    """Train sentiment analysis models"""
    X = df['review']
    y = df['sentiment']
    
    # CountVectorizer
    vectorizer_count = CountVectorizer(max_features=3000)
    X_counts = vectorizer_count.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=42)
    
    # TF-IDF
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

# Header
st.markdown("# 🎬 Movie Review Sentiment Analyzer")
st.markdown("### *IMDB Review Classification System*")
st.markdown("---")

# Top control panel
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    sample_size = st.selectbox(
        "📊 Training Sample Size",
        options=[5000, 10000, 15000, 20000],
        index=1,
        help="Smaller samples train faster"
    )

with col2:
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training in progress..."):
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

with col3:
    if st.session_state.df is not None:
        st.metric("Dataset", f"{len(st.session_state.df):,}")

if not st.session_state.models_trained:
    st.info("👆 Click **'Train Models'** to get started!")

st.markdown("---")

# Main content - Tabs
if st.session_state.models_trained:
    tabs = st.tabs(["🏠 Home", "🔮 Predict", "📊 Analytics", "📈 Insights", "ℹ️ About"])
    
    # Tab 1: Home
    with tabs[0]:
        st.markdown("## Welcome to Sentiment Analysis Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='card' style='text-align: center;'>
                <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>🤖</h2>
                <h3 style='color: white; margin: 0.5rem 0;'>3 ML Models</h3>
                <p style='color: #90caf9; margin: 0;'>Ensemble prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='card' style='text-align: center;'>
                <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>⚡</h2>
                <h3 style='color: white; margin: 0.5rem 0;'>Real-time</h3>
                <p style='color: #90caf9; margin: 0;'>Instant analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='card' style='text-align: center;'>
                <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>📊</h2>
                <h3 style='color: white; margin: 0.5rem 0;'>High Accuracy</h3>
                <p style='color: #90caf9; margin: 0;'>90%+ precision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='card' style='text-align: center;'>
                <h2 style='color: #64b5f6; margin: 0; font-size: 2.5rem;'>☁️</h2>
                <h3 style='color: white; margin: 0.5rem 0;'>Word Clouds</h3>
                <p style='color: #90caf9; margin: 0;'>Visual insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dataset overview
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Positive", f"{(df['sentiment']==1).sum():,}", delta="😊")
        col3.metric("Negative", f"{(df['sentiment']==0).sum():,}", delta="😞")
        col4.metric("Avg Words", f"{df['review_length'].mean():.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Performance Summary
        st.markdown("## 🏆 Model Performance Summary")
        
        results = st.session_state.results
        
        col1, col2, col3 = st.columns(3)
        
        best_model = results.loc[results['Accuracy'].idxmax()]
        
        with col1:
            st.markdown(f"""
            <div class='card'>
                <h3 style='color: #64b5f6; text-align: center;'>🥇 Best Model</h3>
                <h2 style='color: white; text-align: center; margin: 1rem 0;'>{best_model['Model']}</h2>
                <p style='color: #90caf9; text-align: center; font-size: 1.5rem; margin: 0;'>{best_model['Accuracy']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='card'>
                <h3 style='color: #64b5f6; text-align: center;'>📈 F1-Score</h3>
                <h2 style='color: white; text-align: center; margin: 1rem 0;'>Average</h2>
                <p style='color: #90caf9; text-align: center; font-size: 1.5rem; margin: 0;'>{results['F1-Score'].mean():.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='card'>
                <h3 style='color: #64b5f6; text-align: center;'>🎯 ROC-AUC</h3>
                <h2 style='color: white; text-align: center; margin: 1rem 0;'>Average</h2>
                <p style='color: #90caf9; text-align: center; font-size: 1.5rem; margin: 0;'>{results['ROC-AUC'].mean():.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Predict
    with tabs[1]:
        st.markdown("## 🔮 Sentiment Prediction")
        
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Type Review", "📄 Sample Reviews"],
            horizontal=True
        )
        
        if input_method == "✍️ Type Review":
            user_review = st.text_area(
                "Enter movie review:",
                height=150,
                placeholder="Type or paste your movie review here..."
            )
            
            if st.button("🔍 Analyze Sentiment", type="primary"):
                if user_review.strip():
                    with st.spinner("Analyzing..."):
                        processed_review = preprocess_text(user_review)
                        
                        predictions = {}
                        X_input = st.session_state.vectorizer_count.transform([processed_review])
                        predictions['Naive Bayes'] = st.session_state.models['Naive Bayes'].predict(X_input)[0]
                        predictions['Logistic Regression'] = st.session_state.models['Logistic Regression'].predict(X_input)[0]
                        
                        X_input_tfidf = st.session_state.vectorizer_tfidf.transform([processed_review])
                        predictions['SVM (TF-IDF)'] = st.session_state.models['SVM (TF-IDF)'].predict(X_input_tfidf)[0]
                        
                        votes = list(predictions.values())
                        final_prediction = 1 if sum(votes) >= 2 else 0
                        confidence = (sum(votes) / len(votes)) * 100 if final_prediction == 1 else ((len(votes) - sum(votes)) / len(votes)) * 100
                        
                        st.markdown("---")
                        
                        if final_prediction == 1:
                            st.markdown("""
                            <div class='success-box' style='text-align: center;'>
                                <h1 style='color: white; margin: 0; font-size: 3rem;'>😊 POSITIVE</h1>
                                <h3 style='color: #c8e6c9; margin: 1rem 0 0 0;'>This review expresses positive sentiment</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class='error-box' style='text-align: center;'>
                                <h1 style='color: white; margin: 0; font-size: 3rem;'>😞 NEGATIVE</h1>
                                <h3 style='color: #ffcdd2; margin: 1rem 0 0 0;'>This review expresses negative sentiment</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Confidence", f"{confidence:.1f}%")
                        col2.metric("Models Agreeing", f"{max(sum(votes), len(votes)-sum(votes))}/3")
                        col3.metric("Word Count", len(processed_review.split()))
                        
                        with st.expander("📊 Model Predictions Breakdown"):
                            pred_df = pd.DataFrame([
                                {'Model': model, 
                                 'Prediction': '😊 Positive' if pred == 1 else '😞 Negative',
                                 'Value': pred}
                                for model, pred in predictions.items()
                            ])
                            
                            fig = px.bar(
                                pred_df, x='Model', y='Value',
                                color='Prediction',
                                color_discrete_map={'😊 Positive': '#4caf50', '😞 Negative': '#f44336'},
                                title='Individual Model Predictions'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#e3f2fd'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Please enter a review to analyze.")
        
        else:  # Sample Reviews
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 😞 Sample Negative Reviews")
                neg_samples = df[df['sentiment'] == 0].sample(3, random_state=42)
                for idx, (i, row) in enumerate(neg_samples.iterrows()):
                    if st.button(f"Negative Sample {idx+1}", key=f"neg_{i}"):
                        st.session_state.selected_sample = row['review']
                        st.session_state.selected_label = 0
            
            with col2:
                st.markdown("#### 😊 Sample Positive Reviews")
                pos_samples = df[df['sentiment'] == 1].sample(3, random_state=42)
                for idx, (i, row) in enumerate(pos_samples.iterrows()):
                    if st.button(f"Positive Sample {idx+1}", key=f"pos_{i}"):
                        st.session_state.selected_sample = row['review']
                        st.session_state.selected_label = 1
            
            if 'selected_sample' in st.session_state:
                st.markdown("---")
                st.markdown("#### Selected Review:")
                st.text_area("", st.session_state.selected_sample[:500] + "...", height=100, disabled=True)
                
                if st.button("🔍 Analyze This Review", type="primary"):
                    with st.spinner("Analyzing..."):
                        processed_review = st.session_state.selected_sample
                        
                        predictions = {}
                        X_input = st.session_state.vectorizer_count.transform([processed_review])
                        predictions['Naive Bayes'] = st.session_state.models['Naive Bayes'].predict(X_input)[0]
                        predictions['Logistic Regression'] = st.session_state.models['Logistic Regression'].predict(X_input)[0]
                        
                        X_input_tfidf = st.session_state.vectorizer_tfidf.transform([processed_review])
                        predictions['SVM (TF-IDF)'] = st.session_state.models['SVM (TF-IDF)'].predict(X_input_tfidf)[0]
                        
                        votes = list(predictions.values())
                        final_prediction = 1 if sum(votes) >= 2 else 0
                        confidence = (sum(votes) / len(votes)) * 100 if final_prediction == 1 else ((len(votes) - sum(votes)) / len(votes)) * 100
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Prediction:")
                            if final_prediction == 1:
                                st.success("😊 POSITIVE")
                            else:
                                st.error("😞 NEGATIVE")
                        
                        with col2:
                            st.markdown("### Actual:")
                            if st.session_state.selected_label == 1:
                                st.success("😊 POSITIVE")
                            else:
                                st.error("😞 NEGATIVE")
                        
                        if final_prediction == st.session_state.selected_label:
                            st.success("🎯 Correct Prediction!")
                        else:
                            st.warning("❌ Incorrect Prediction")
                        
                        st.metric("Confidence", f"{confidence:.1f}%")
    
    # Tab 3: Analytics
    with tabs[2]:
        st.markdown("## 📊 Model Performance Analytics")
        
        results = st.session_state.results
        
        # Metrics Overview
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
        
        # Performance Table
        st.markdown("### 📋 Detailed Metrics")
        styled_results = results.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}'
        }).background_gradient(cmap='Blues', subset=['Accuracy', 'F1-Score'])
        
        st.dataframe(styled_results, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['#1565c0', '#1976d2', '#42a5f5', '#64b5f6']
            
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
                title='Model Performance Comparison',
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd',
                xaxis_title='Model',
                yaxis_title='Score'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart - Navy blue theme
            fig = go.Figure()
            
            navy_colors = ['#1565c0', '#1976d2', '#42a5f5']
            
            for idx, row in results.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    fill='toself',
                    name=row['Model'],
                    line_color=navy_colors[idx % len(navy_colors)],
                    fillcolor=navy_colors[idx % len(navy_colors)],
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor='rgba(21, 101, 192, 0.3)'
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(21, 101, 192, 0.3)'
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                title='Model Performance Radar',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Insights
    with tabs[3]:
        st.markdown("## 📈 Data Insights & Visualizations")
        
        df = st.session_state.df
        
        # Word Clouds
        st.markdown("### ☁️ Word Cloud Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 😞 Negative Reviews")
            negative_text = ' '.join(df[df['sentiment'] == 0]['review'])
            wordcloud_neg = WordCloud(
                width=800, height=400,
                background_color='#0a1628',
                colormap='Reds',
                max_words=100
            ).generate(negative_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud_neg, interpolation='bilinear')
            ax.axis('off')
            ax.set_facecolor('#0a1628')
            fig.patch.set_facecolor('#0a1628')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 😊 Positive Reviews")
            positive_text = ' '.join(df[df['sentiment'] == 1]['review'])
            wordcloud_pos = WordCloud(
                width=800, height=400,
                background_color='#0a1628',
                colormap='Greens',
                max_words=100
            ).generate(positive_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud_pos, interpolation='bilinear')
            ax.axis('off')
            ax.set_facecolor('#0a1628')
            fig.patch.set_facecolor('#0a1628')
            st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Review Length Distribution
        st.markdown("### 📏 Review Length Distribution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df[df['sentiment'] == 0]['review_length'],
            name='Negative',
            opacity=0.7,
            marker_color='#f44336',
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=df[df['sentiment'] == 1]['review_length'],
            name='Positive',
            opacity=0.7,
            marker_color='#4caf50',
            nbinsx=50
        ))
        
        fig.update_layout(
            barmode='overlay',
            title='Word Count Distribution: Positive vs Negative Reviews',
            xaxis_title='Number of Words',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e3f2fd'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Sentiment Distribution")
            labels = ['Negative 😞', 'Positive 😊']
            values = [(df['sentiment']==0).sum(), (df['sentiment']==1).sum()]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=['#f44336', '#4caf50']
            )])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Statistical Summary")
            summary_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Negative': [
                    df[df['sentiment']==0]['review_length'].mean(),
                    df[df['sentiment']==0]['review_length'].median(),
                    df[df['sentiment']==0]['review_length'].std(),
                    df[df['sentiment']==0]['review_length'].min(),
                    df[df['sentiment']==0]['review_length'].max()
                ],
                'Positive': [
                    df[df['sentiment']==1]['review_length'].mean(),
                    df[df['sentiment']==1]['review_length'].median(),
                    df[df['sentiment']==1]['review_length'].std(),
                    df[df['sentiment']==1]['review_length'].min(),
                    df[df['sentiment']==1]['review_length'].max()
                ]
            })
            
            summary_df['Negative'] = summary_df['Negative'].round(2)
            summary_df['Positive'] = summary_df['Positive'].round(2)
            
            st.dataframe(summary_df, use_container_width=True)
    
    # Tab 5: About
    with tabs[4]:
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        This **Sentiment Analysis Dashboard** analyzes IMDB movie reviews to classify them as positive or negative
        using advanced machine learning techniques and natural language processing.
        
        ## 🔬 Technical Details
        
        ### Machine Learning Models:
        - **Naive Bayes**: Probabilistic classifier perfect for text classification
        - **Logistic Regression**: Linear model with high interpretability
        - **Support Vector Machine**: Maximum-margin classifier with TF-IDF features
        
        ### NLP Pipeline:
        1. **HTML Removal**: Clean HTML tags from text
        2. **Lowercasing**: Normalize text case
        3. **Special Character Removal**: Keep only letters
        4. **Stopword Removal**: Remove common words
        5. **Lemmatization**: Reduce words to base form
        6. **Vectorization**: Convert to numerical features (Count & TF-IDF)
        
        ### Dataset:
        - Source: IMDB Movie Reviews Dataset
        - Total Reviews: 50,000
        - Classes: Binary (Positive/Negative)
        - Balanced dataset
        
        ## 📊 Performance Metrics
        
        - **Accuracy**: Overall correctness
        - **Precision**: Correct positive predictions
        - **Recall**: Coverage of actual positives
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC-AUC**: Discrimination ability
        
        ## 🛠️ Technologies
        
        - **Frontend**: Streamlit
        - **ML**: Scikit-learn
        - **NLP**: NLTK
        - **Visualization**: Plotly, Matplotlib, WordCloud
        - **Data**: Pandas, NumPy
        
        ## 👨‍💻 Developer
        
        Created as part of NLP Internship Projects
        
        ## 📝 Note
        
        This system is for educational purposes. Sentiment analysis provides general insights
        but may not capture nuanced opinions in all cases.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #64b5f6; font-size: 1rem;'>
            <p>© 2026 Sentiment Analysis Dashboard | Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen when models not trained
    st.markdown("""
    <div class='card' style='text-align: center; padding: 3rem;'>
        <h2 style='color: #64b5f6; margin: 0;'>🎬 Welcome to Sentiment Analyzer</h2>
        <p style='color: #90caf9; font-size: 1.2rem; margin: 2rem 0;'>
            Analyze movie reviews with sentiment detection
        </p>
        <p style='color: #e3f2fd; margin: 0;'>
            👆 Click <strong>'Train Models'</strong> above to get started
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64b5f6; padding: 1rem;'>
    <p>💡 Powered by Machine Learning | Real-time Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)
