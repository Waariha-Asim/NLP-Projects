import streamlit as st
import pandas as pd
import json
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
import torch
from transformers import pipeline

# Check if CUDA is available
TORCH_AVAILABLE = True
CUDA_AVAILABLE = torch.cuda.is_available()

# Page Configuration
st.set_page_config(
    page_title="Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Modern Black Theme with Glassmorphism & Purple/Teal Accents
st.markdown("""
<style>
    /* Main background - Pure Black with gradient accents */
    .stApp {
        background: #000000;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(20, 184, 166, 0.06) 0%, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Sidebar - Glassmorphism with purple/teal accent */
    [data-testid="stSidebar"] {
        background: rgba(10, 5, 20, 0.85);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(124, 58, 237, 0.3);
        box-shadow: 4px 0 24px rgba(124, 58, 237, 0.15);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
        text-shadow: 0 0 20px rgba(167, 139, 250, 0.4);
    }
    
    /* Headers - Elegant Purple/Teal with Glow */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-align: center;
        letter-spacing: -1px !important;
        text-shadow: 0 0 40px rgba(124, 58, 237, 0.6);
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #7c3aed, #14b8a6, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #c4b5fd !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 0 15px rgba(196, 181, 253, 0.3);
    }
    
    h3 {
        color: #5eead4 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons - Purple/Teal Glassmorphism */
    .stButton>button {
        background: rgba(124, 58, 237, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #a78bfa;
        border: 2px solid rgba(167, 139, 250, 0.4);
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        transition: all 0.4s ease;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background: rgba(124, 58, 237, 0.4);
        border-color: #7c3aed;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.5);
        transform: translateY(-3px) scale(1.02);
    }
    
    /* Text Input - Glassmorphism */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div {
        background: rgba(20, 10, 40, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #f5f3ff;
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 10px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stTextInput>div>div>input:focus, 
    .stTextArea>div>div>textarea:focus {
        border-color: #7c3aed;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
        background: rgba(20, 10, 40, 0.8);
    }
    
    /* Metrics - Glowing Purple/Teal */
    [data-testid="stMetricValue"] {
        color: #7c3aed !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 25px rgba(124, 58, 237, 0.7);
    }
    
    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Info/Success/Warning boxes - Glassmorphism */
    .stAlert {
        background: rgba(124, 58, 237, 0.2);
        backdrop-filter: blur(10px);
        border-left: 4px solid #7c3aed;
        border-radius: 12px;
        color: #ddd6fe;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.2);
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #c4b5fd !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Dataframe - Glassmorphism */
    .dataframe {
        background: rgba(20, 10, 40, 0.7) !important;
        backdrop-filter: blur(10px);
        color: #f5f3ff !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 10px;
    }
    
    /* Expander - Glassmorphism */
    .streamlit-expanderHeader {
        background: rgba(124, 58, 237, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        color: #a78bfa !important;
        font-weight: 700;
        border: 1px solid rgba(167, 139, 250, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(124, 58, 237, 0.25);
        border-color: #7c3aed;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.2);
    }
    
    /* Cards - Glassmorphism */
    .card {
        background: rgba(20, 10, 40, 0.5);
        backdrop-filter: blur(15px) saturate(180%);
        -webkit-backdrop-filter: blur(15px) saturate(180%);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(124, 58, 237, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 12px 40px rgba(124, 58, 237, 0.3);
        transform: translateY(-5px);
    }
    
    .answer-box {
        background: rgba(20, 10, 40, 0.6);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
        transition: all 0.3s ease;
    }
    
    .answer-box:hover {
        border-color: #7c3aed;
        box-shadow: 0 12px 40px rgba(124, 58, 237, 0.4);
    }
    
    /* Scrollbar - Purple/Teal Theme */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.4);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7c3aed 0%, #14b8a6 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #8b5cf6 0%, #2dd4bf 100%);
    }
    
    /* Links - Purple Glow */
    a {
        color: #a78bfa !important;
        text-decoration: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    a:hover {
        color: #c4b5fd !important;
        text-shadow: 0 0 15px rgba(167, 139, 250, 0.6);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed 0%, #14b8a6 100%);
    }
    
    /* Tabs - Glassmorphism */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(20, 10, 40, 0.4);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #c4b5fd;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.3);
        backdrop-filter: blur(10px);
        color: #ffffff;
        border: 1px solid rgba(167, 139, 250, 0.5);
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #c4b5fd !important;
        font-weight: 700;
        font-size: 1.05rem;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_pipeline' not in st.session_state:
    st.session_state.qa_pipeline = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Model configurations
MODELS = {
    'DistilBERT (Fast)': {
        'id': 'distilbert-base-cased-distilled-squad',
        'description': 'Lightweight and fast, 40% smaller than BERT',
        'emoji': '⚡'
    },
    'RoBERTa (Robust)': {
        'id': 'deepset/roberta-base-squad2',
        'description': 'Robustly optimized BERT, handles unanswerable questions',
        'emoji': '🛡️'
    }
}

@st.cache_resource
def load_qa_model(model_name):
    """Load question answering model"""
    model_id = MODELS[model_name]['id']
    # Use GPU if available, otherwise CPU
    device = 0 if CUDA_AVAILABLE else -1
    
    with st.spinner(f"Loading {model_name}..."):
        qa_pipeline = pipeline(
            'question-answering',
            model=model_id,
            tokenizer=model_id,
            device=device
        )
    return qa_pipeline

def answer_question(context, question, qa_pipeline):
    """Get answer from QA model"""
    start_time = time.time()
    
    result = qa_pipeline({
        'question': question,
        'context': context
    })
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        'answer': result['answer'],
        'score': result['score'],
        'start': result['start'],
        'end': result['end'],
        'inference_time': inference_time
    }

# Sidebar
with st.sidebar:
    st.markdown("# 🤖Q&A System")
    st.markdown("### *Powered by Transformers*")
    st.markdown("---")
    
    st.markdown("### ⚙️ Model Selection")
    
    selected_model = st.selectbox(
        "Choose Model",
        options=list(MODELS.keys()),
        index=0,
        help="Select transformer model for question answering"
    )
    
    # Display model info
    model_info = MODELS[selected_model]
    st.markdown(f"""
    <div class='card' style='text-align: center;'>
        <h2 style='color: #a78bfa; margin: 0; font-size: 2.5rem;'>{model_info['emoji']}</h2>
        <p style='color: #c4b5fd; margin: 1rem 0 0 0; font-size: 0.9rem;'>{model_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 LOAD MODEL", use_container_width=True):
        st.session_state.qa_pipeline = load_qa_model(selected_model)
        st.session_state.current_model = selected_model
        st.success(f"✅ {selected_model} loaded!")
    
    st.markdown("---")
    
    # System info
    st.markdown("### 💻 System Info")
    
    if st.session_state.current_model:
        st.success(f"✅ **Active:** {st.session_state.current_model}")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.history:
        st.markdown("### 📊 Session Stats")
        st.metric("Questions Asked", len(st.session_state.history))
        avg_confidence = np.mean([h['score'] for h in st.session_state.history]) * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Header
st.markdown("# 🤖 Question Answering System")
st.markdown("### *Extract Answers from Text using Transformers*")
st.markdown("---")

# Main content
if st.session_state.qa_pipeline is None:
    # Welcome screen
    st.markdown("""
    <div class='answer-box' style='text-align: center; padding: 3rem;'>
        <h2 style='color: #a78bfa; margin: 0;'>🤖 Welcome to Question Answering</h2>
        <p style='color: #c4b5fd; font-size: 1.3rem; margin: 2rem 0;'>
            Ask questions and get instant answers from any text passage
        </p>
        <p style='color: #5eead4; font-size: 1.1rem; margin: 0;'>
            ⚡ <strong>DistilBERT</strong> | 🛡️ <strong>RoBERTa</strong>
        </p>
        <p style='color: #7c3aed; font-size: 1.2rem; margin: 2rem 0 0 0; font-weight: 700;'>
            👈 Select a model in the sidebar and click <strong>'LOAD MODEL'</strong> to begin
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #a78bfa; margin: 0; font-size: 2.5rem;'>🎯</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>Accurate</h3>
            <p style='color: #c4b5fd; margin: 0;'>Precise Answer Extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #14b8a6; margin: 0; font-size: 2.5rem;'>⚡</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>Fast</h3>
            <p style='color: #5eead4; margin: 0;'>Real-time Inference</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #a78bfa; margin: 0; font-size: 2.5rem;'>🤖</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>2 Models</h3>
            <p style='color: #c4b5fd; margin: 0;'>Compare Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h2 style='color: #14b8a6; margin: 0; font-size: 2.5rem;'>🧠</h2>
            <h3 style='color: white; margin: 0.5rem 0;'>Transformers</h3>
            <p style='color: #5eead4; margin: 0;'>State-of-the-art NLP</p>
        </div>
        """, unsafe_allow_html=True)
    
else:
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Ask Question", "📚 Examples", "📊 History", "ℹ️ About"])
    
    # Tab 1: Ask Question
    with tab1:
        st.markdown("## 🔍 Ask Your Question")
        
        # Context input
        st.markdown("### 📄 Context (Passage)")
        context = st.text_area(
            "",
            height=200,
            placeholder="Paste the text passage here...\n\nExample: 'The Eiffel Tower is located in Paris, France. It was designed by engineer Gustave Eiffel and completed in 1889...'",
            key="context_input"
        )
        
        # Question input
        st.markdown("### ❓ Question")
        question = st.text_input(
            "",
            placeholder="What do you want to know? (e.g., 'Where is the Eiffel Tower located?')",
            key="question_input"
        )
        
        # Ask button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ask_button = st.button("🔍 GET ANSWER", use_container_width=True, type="primary")
        
        if ask_button:
            if not context.strip():
                st.warning("⚠️ Please provide a context passage.")
            elif not question.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("🤔 Finding answer..."):
                    result = answer_question(context, question, st.session_state.qa_pipeline)
                    
                    # Add to history
                    st.session_state.history.append({
                        'context': context,
                        'question': question,
                        'answer': result['answer'],
                        'score': result['score'],
                        'inference_time': result['inference_time'],
                        'model': st.session_state.current_model
                    })
                    
                    # Display answer
                    st.markdown("---")
                    st.markdown("## ✨ Answer")
                    
                    confidence_color = "#10b981" if result['score'] > 0.7 else "#f59e0b" if result['score'] > 0.4 else "#ef4444"
                    confidence_emoji = "🟢" if result['score'] > 0.7 else "🟡" if result['score'] > 0.4 else "🔴"
                    
                    st.markdown(f"""
                    <div class='answer-box'>
                        <h2 style='color: white; margin: 0 0 1rem 0; font-size: 1.8rem;'>{result['answer']}</h2>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: #c4b5fd; font-weight: 600;'>Confidence: </span>
                                <span style='color: {confidence_color}; font-weight: 700; font-size: 1.2rem;'>{confidence_emoji} {result['score']*100:.2f}%</span>
                            </div>
                            <div>
                                <span style='color: #c4b5fd; font-weight: 600;'>Time: </span>
                                <span style='color: #5eead4; font-weight: 700;'>{result['inference_time']:.2f} ms</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show context with highlighted answer
                    with st.expander("📖 View Context with Answer Highlighted"):
                        highlighted_context = (
                            context[:result['start']] +
                            f"**<span style='background-color: rgba(124, 58, 237, 0.3); padding: 2px 4px; border-radius: 4px;'>{result['answer']}</span>**" +
                            context[result['end']:]
                        )
                        st.markdown(highlighted_context, unsafe_allow_html=True)
    
    # Tab 2: Examples
    with tab2:
        st.markdown("## 📚 Pre-loaded Examples")
        st.markdown("Click on any example to try it out!")
        
        examples = [
            {
                'title': '🌍 Geography - Amazon Rainforest',
                'context': "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%.",
                'questions': [
                    "Which country has the most rainforest?",
                    "How large is the Amazon basin?",
                    "What percentage does Peru have?"
                ]
            },
            {
                'title': '💻 Technology - Python Programming',
                'context': "Python is a high-level, interpreted programming language with dynamic semantics. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
                'questions': [
                    "Who created Python?",
                    "When was Python first released?",
                    "What does Python's design emphasize?"
                ]
            },
            {
                'title': '🗼 History - Eiffel Tower',
                'context': "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.",
                'questions': [
                    "Who was the Eiffel Tower named after?",
                    "When was the Eiffel Tower constructed?",
                    "Where is the Eiffel Tower located?"
                ]
            },
            {
                'title': '🤖 Machine Learning',
                'context': "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future.",
                'questions': [
                    "What is machine learning a subset of?",
                    "Does machine learning require explicit programming?",
                    "What does the learning process begin with?"
                ]
            },
            {
                'title': '🌌 Science - Solar System',
                'context': "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It was formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. The four smaller inner system planets, Mercury, Venus, Earth and Mars, are terrestrial planets, being composed primarily of rock and metal.",
                'questions': [
                    "How old is the Solar System?",
                    "Where is most of the system's mass?",
                    "What are the inner planets composed of?"
                ]
            }
        ]
        
        for idx, example in enumerate(examples):
            with st.expander(f"{example['title']}"):
                st.markdown(f"**Context:** {example['context'][:150]}...")
                st.markdown("**Suggested Questions:**")
                
                for q_idx, q in enumerate(example['questions']):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"{q_idx + 1}. {q}")
                    with col2:
                        if st.button("Try", key=f"example_{idx}_{q_idx}"):
                            with st.spinner("🤔 Finding answer..."):
                                result = answer_question(example['context'], q, st.session_state.qa_pipeline)
                                
                                st.session_state.history.append({
                                    'context': example['context'],
                                    'question': q,
                                    'answer': result['answer'],
                                    'score': result['score'],
                                    'inference_time': result['inference_time'],
                                    'model': st.session_state.current_model
                                })
                                
                                st.success(f"**Answer:** {result['answer']} (Confidence: {result['score']*100:.1f}%)")
    
    # Tab 3: History
    with tab3:
        st.markdown("## 📊 Question History")
        
        if not st.session_state.history:
            st.info("📭 No questions asked yet. Start asking questions in the 'Ask Question' tab!")
        else:
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Questions", len(st.session_state.history))
            col2.metric("Avg Confidence", f"{np.mean([h['score'] for h in st.session_state.history])*100:.1f}%")
            col3.metric("Avg Response Time", f"{np.mean([h['inference_time'] for h in st.session_state.history]):.1f} ms")
            
            high_conf = sum(1 for h in st.session_state.history if h['score'] > 0.7)
            col4.metric("High Confidence", f"{high_conf}/{len(st.session_state.history)}")
            
            st.markdown("---")
            
            # Confidence distribution chart
            st.markdown("### 📈 Confidence Score Distribution")
            
            confidence_scores = [h['score'] * 100 for h in st.session_state.history]
            
            fig = go.Figure(data=[go.Histogram(
                x=confidence_scores,
                nbinsx=20,
                marker_color='rgba(124, 58, 237, 0.7)',
                marker_line_color='#7c3aed',
                marker_line_width=2
            )])
            
            fig.update_layout(
                xaxis_title='Confidence Score (%)',
                yaxis_title='Frequency',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#c4b5fd',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # History table
            st.markdown("### 📋 Question History")
            
            history_df = pd.DataFrame([{
                'Model': h['model'],
                'Question': h['question'][:50] + '...' if len(h['question']) > 50 else h['question'],
                'Answer': h['answer'],
                'Confidence': f"{h['score']*100:.1f}%",
                'Time (ms)': f"{h['inference_time']:.2f}"
            } for h in reversed(st.session_state.history)])
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Tab 4: About
    with tab4:
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        This **Question Answering System** uses state-of-the-art transformer models to extract precise answers from text passages.
        
        ## 📊 Dataset
        
        **SQuAD (Stanford Question Answering Dataset) v1.1**
        - 📥 [Download from Kaggle](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset)
        - Contains 100,000+ question-answer pairs
        - Based on Wikipedia articles
        - Human-annotated answers
        
        ## 🤖 Transformer Models
        
        ### 1. **DistilBERT** ⚡
        - **Speed**: Fastest (40% smaller than BERT)
        - **Accuracy**: Good
        - **Best for**: Real-time applications, quick responses
        
        ### 2. **RoBERTa** 🛡️
        - **Speed**: Medium
        - **Accuracy**: Excellent
        - **Best for**: Robust performance, handling edge cases
        
        ## 📈 Evaluation Metrics
        
        - **Exact Match (EM)**: Percentage of predictions that match ground truth exactly
        - **F1 Score**: Token-level overlap between prediction and ground truth
        - **Confidence Score**: Model's confidence in the predicted answer (0-1)
        - **Inference Time**: Time taken to generate answer (milliseconds)
        
        ## 🔬 How It Works
        
        1. **Input**: Context passage + Question
        2. **Tokenization**: Convert text to tokens the model understands
        3. **Encoding**: Transform tokens into numerical representations
        4. **Span Extraction**: Model identifies start and end positions of answer
        5. **Decoding**: Convert token positions back to text
        6. **Output**: Extracted answer with confidence score
        
        ## 🛠️ Technologies
        
        - **Framework**: Streamlit (Web Interface)
        - **Models**: Hugging Face Transformers
        - **Deep Learning**: PyTorch
        - **Visualization**: Plotly
        - **Data Processing**: Pandas, NumPy
        
        ## 📚 Use Cases
        
        - 📖 **Reading Comprehension**: Educational tools
        - 🔍 **Information Retrieval**: Search systems
        - 💬 **Customer Support**: Automated FAQ responses
        - 📝 **Document Analysis**: Extract key information
        - 🎓 **Research**: Academic paper analysis
        
        ## 🚀 Features
        
        ✅ Multiple transformer models for comparison  
        ✅ Real-time answer extraction  
        ✅ Confidence scoring  
        ✅ Answer highlighting in context  
        ✅ Question history tracking  
        ✅ Performance analytics  
        ✅ Pre-loaded example passages  
        ✅ Beautiful modern UI with glassmorphism  
        
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #a78bfa; padding: 1rem;'>
    <p style='font-size: 1.1rem;'>© 2026 Question Answering System | Powered by Transformers 🤖</p>
</div>
""", unsafe_allow_html=True)
