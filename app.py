import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="Huggingface_AI",
    page_icon=None,
    layout="wide"
)

# Simple CSS Styling
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #eef2ff, #f8fafc); }
h1, h2, h3 { color: #312e81; }
.stButton button { background-color: #4338ca; color: white; border-radius: 8px; border: none; padding: 8px 20px; }
.stButton button:hover { background-color: #3730a3; color: white; }
</style>
""", unsafe_allow_html=True)

# Load AI Models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_translator():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

# Application Header
st.title("Huggingface_AI")
st.write("AI Text Processing using Hugging Face Pipelines")
st.divider()

# Sidebar
task = st.sidebar.selectbox(
    "Select an AI Task",
    [
        "Home",
        "Summarization",
        "Translation",
        "Sentiment Analysis",
        "Question Answering"
    ]
)

# Home Page
if task == "Home":
    st.header("Welcome to Huggingface_AI")
    st.write("This application demonstrates four AI tasks using pre-trained Hugging Face models.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Summarization")
        st.write("Convert long text into a short summary.")
        st.subheader("2. Translation")
        st.write("Translate English text into French.")
    with col2:
        st.subheader("3. Sentiment Analysis")
        st.write("Identify whether text is positive or negative.")
        st.subheader("4. Question Answering")
        st.write("Answer questions using the given context.")
        
    st.info("Select a task from the sidebar to get started.")

# Summarization
elif task == "Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter text to summarize", height=200, placeholder="Enter a paragraph here...")
    
    if st.button("Summarize"):
        if text.strip():
            try:
                with st.spinner("Generating summary..."):
                    summarizer = load_summarizer()
                    result = summarizer(text, max_length=100, min_length=20, do_sample=False, truncation=True)
                st.subheader("Summary")
                st.success(result[0]["summary_text"])
            except Exception as e:
                st.error("Unable to summarize. Check your internet connection.")
                st.caption(str(e))
        else:
            st.warning("Please enter some text.")

# Translation
elif task == "Translation":
    st.header("English to French Translation")
    text = st.text_area("Enter English text", height=150, placeholder="Enter English text here...")
    
    if st.button("Translate"):
        if text.strip():
            try:
                with st.spinner("Translating text..."):
                    translator = load_translator()
                    result = translator(text, max_length=200)
                st.subheader("French Translation")
                st.success(result[0]["translation_text"])
            except Exception as e:
                st.error("Unable to translate. Check your internet connection.")
                st.caption(str(e))
        else:
            st.warning("Please enter some text.")

# Sentiment Analysis
elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter a sentence", height=120, placeholder="Example: I really enjoyed this application.")
    
    if st.button("Analyze Sentiment"):
        if text.strip():
            try:
                with st.spinner("Analyzing sentiment..."):
                    sentiment_model = load_sentiment()
                    result = sentiment_model(text)
                    label = result[0]["label"]
                    score = result[0]["score"]
                st.subheader("Analysis Result")
                st.write("Sentiment:", label)
                st.write("Confidence:", f"{score * 100:.2f}%")
            except Exception as e:
                st.error("Unable to analyze sentiment. Check your internet connection.")
                st.caption(str(e))
        else:
            st.warning("Please enter a sentence.")

# Question Answering
elif task == "Question Answering":
    st.header("Question Answering")
    context = st.text_area("Enter the context", height=200, placeholder="Enter a paragraph containing the answer...")
    question = st.text_input("Ask a question", placeholder="Enter your question here...")
    
    if st.button("Get Answer"):
        if context.strip() and question.strip():
            try:
                with st.spinner("Finding the answer..."):
                    qa_model = load_qa()
                    result = qa_model(question=question, context=context)
                st.subheader("Answer")
                st.success(result["answer"])
            except Exception as e:
                st.error("Unable to answer the question. Check your internet connection.")
                st.caption(str(e))
        else:
            st.warning("Please enter both context and question.")

# Footer
st.divider()
st.caption("Huggingface_AI | Built with Python, Streamlit and Hugging Face")
