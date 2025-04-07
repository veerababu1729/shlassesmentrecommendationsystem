import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Config
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

# Title
st.title("🔍 SHL Assessment Recommendation System")
st.write("Enter a job description or query to get best-matched SHL assessments.")

# Gemini API setup (replace with your own API key)
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-pro")

# Load SHL assessments
df = pd.read_csv("data/assessments.csv")  # make sure this file exists

# Get user input
query = st.text_area("Paste job description or query here:")

# Embed using Gemini
def get_embedding(text):
    response = model.embed_content(
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

# Run Recommendation
if st.button("Recommend"):
    with st.spinner("Analyzing with Gemini..."):
        try:
            query_vec = get_embedding(query)
            df["score"] = df["full_text"].apply(lambda x: cosine_similarity([query_vec], [get_embedding(x)])[0][0])
            top_df = df.sort_values("score", ascending=False).head(10)

            st.success("Top Recommended Assessments:")
            st.dataframe(top_df[["Assessment Name", "URL", "Duration", "Remote Testing Support", "Adaptive/IRT Support", "Test Type"]])

        except Exception as e:
            st.error(f"Error during recommendation: {e}")
