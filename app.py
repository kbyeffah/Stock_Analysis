import faiss
import numpy as np
import yfinance as yf
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Custom Styling
st.markdown(
    """
    <style>
        body {background-color: #0A192F; color: #E0E5EC; font-family: 'Segoe UI', sans-serif;}
        .stTextInput > div > div > input {
            background-color: #112240; 
            color: #E0E5EC; 
            border-radius: 8px; 
            padding: 10px; 
            border: 2px solid transparent;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);
        }
        .stButton > button {
            background-color: #1F4068; 
            color: #E0E5EC; 
            border: none; 
            border-radius: 8px; 
            padding: 12px 18px; 
            transition: 0.3s;
            font-weight: bold;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stButton > button:hover {
            background-color: #2C698D;
            transform: scale(1.05);
        }
        .stTitle {color: white; font-size: 28px; font-weight: bold;}
        .container {
            background-color: #112240;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit UI elements
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

# Load Sentence-Transformer Model (22MB)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

def get_embeddings(text):
    """Fetch embeddings using the local MiniLM model."""
    return np.array(embedding_model.encode(text))

def get_stock_info(symbol: str) -> dict:
    """Retrieve stock information and description from Yahoo Finance."""
    try:
        data = yf.Ticker(symbol)
        stock_info = data.info
        description = stock_info.get("longBusinessSummary", "No description available.")
        return {"symbol": symbol, "description": description}
    except Exception as e:
        return {"symbol": symbol, "description": f"Error retrieving stock info: {str(e)}"}

# Determine embedding size dynamically
test_embedding = get_embeddings("Test")
d = test_embedding.shape[0]

# Initialize FAISS index
index = faiss.IndexFlatL2(d)
stock_metadata = {}

def store_stock_embeddings(stock_list):
    """Store stock embeddings in FAISS index."""
    global stock_metadata
    vectors = []
    metadata = []

    for stock in stock_list:
        description = stock["description"]
        symbol = stock["symbol"]
        embedding = get_embeddings(description)

        if np.any(embedding):  
            vectors.append(embedding)
            metadata.append({"symbol": symbol, "description": description})

    if vectors:
        index.add(np.array(vectors))
        for i, meta in enumerate(metadata):
            stock_metadata[len(stock_metadata)] = meta

def find_similar_stocks(query):
    """Find similar stocks based on query embedding."""
    if index.ntotal == 0:
        return []

    query_embedding = get_embeddings(query).reshape(1, -1)
    D, I = index.search(query_embedding, k=10)
    return [stock_metadata[idx] for idx in I[0] if idx in stock_metadata]

def analyze_stocks(query, stocks):
    """Generate stock analysis using Groq's Llama model."""
    if not stocks:
        return "No relevant stocks found."

    context = "\n".join([f"Symbol: {s['symbol']}, Description: {s['description']}" for s in stocks])
    prompt = f"""
    You are a financial assistant. Analyze the following stocks based on the given query: {query}.

    Stock data:
    {context}

    Provide insights based on performance, trends, and any notable aspects.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# Load stock data at startup
default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
stock_data = [get_stock_info(symbol) for symbol in default_stocks]
store_stock_embeddings(stock_data)

# Streamlit UI
st.title('Stock Analysis Dashboard')

with st.container():
    query = st.text_input('Ask About Stocks:', '')

    if st.button('Get Stock Info'):
        stocks = find_similar_stocks(query)
        analysis = analyze_stocks(query, stocks)

        st.markdown("### Stock Insights:")
        st.markdown(f"<div class='stMarkdown'>{analysis}</div>", unsafe_allow_html=True)

        st.markdown("---")
        if not stocks:
            st.error("No relevant stocks found.")