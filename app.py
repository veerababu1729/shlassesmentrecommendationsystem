import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os

# Setup Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
embedding_model = genai.EmbeddingModel(model_name="models/embedding-001")

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("🔍 SHL Assessment Recommendation System")
st.write("Paste a job description or query below:")

# Load data
df = pd.read_csv("data/assessments.csv")

# Get user input
query = st.text_area("Job Description or Query")

def get_embedding(text):
    response = embedding_model.embed(content=text, task_type="retrieval_document")
    return np.array(response["embedding"])

# Recommend
if st.button("Recommend"):
    with st.spinner("Analyzing using Gemini..."):
        try:
            query_vec = get_embedding(query)
            df["score"] = df["full_text"].apply(lambda x: cosine_similarity([query_vec], [get_embedding(x)])[0][0])
            top_df = df.sort_values("score", ascending=False).head(10)
            st.success("Top Recommended Assessments:")
            st.dataframe(top_df[["Assessment Name", "URL", "Duration", "Remote Testing Support", "Adaptive/IRT Support", "Test Type"]])
        except Exception as e:
            st.error(f"Error during recommendation: {e}")
