# app.py
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# Set Gemini API key (add in Streamlit Cloud secrets later)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load assessment data
df = pd.read_csv("data/assessments.csv")

# Load embedding model and build index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["description"].tolist())
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Gemini model for summarizing input
gemini = genai.GenerativeModel("gemini-pro")

# UI starts here
st.title("🔍 SHL Assessment Recommender")

user_input = st.text_area("Enter job description or query:", height=150)

if st.button("Recommend Assessments"):
    if not user_input.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Thinking with Gemini..."):
            prompt = f"""Summarize this job description to match assessment test descriptions:\n\n{user_input}\n\nSummary:"""
            summary = gemini.generate_content(prompt).text.strip()

        q_embedding = model.encode([summary])
        D, I = index.search(np.array(q_embedding), 10)
        results = df.iloc[I[0]]

        st.markdown("### 🔗 Top Assessment Matches")
        st.markdown(f"**Gemini Summary**: {summary}")

        for _, row in results.iterrows():
            st.markdown(f"""
            **[{row['name']}]({row['url']})**  
            - ⏱ Duration: {row['duration']}  
            - 📋 Type: {row['test_type']}  
            - 🌐 Remote: {row['remote']}  
            - 🧠 Adaptive: {row['adaptive']}  
            ---
            """)

