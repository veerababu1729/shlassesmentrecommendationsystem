import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os

# Setup Gemini API - Use proper error handling for API key
try:
    # Try to get from environment variables
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # If not found in environment, use st.secrets (Streamlit's way to access secrets)
    if api_key is None:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    
    # If still None, we'll handle it with a user input field
    if api_key is None:
        api_key = st.text_input("Enter your Gemini API Key:", type="password")
        if not api_key:
            st.warning("Please enter a valid API key to use this app.")
            st.stop()
    
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring API: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("🔍 SHL Assessment Recommendation System")
st.write("Paste a job description or query below:")

# Load data
try:
    df = pd.read_csv("data/assessments.csv")
    
    # Add a full_text column if it doesn't exist
    if "full_text" not in df.columns:
        # Create full_text by combining available information
        df["full_text"] = df.apply(
            lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}.", 
            axis=1
        )
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Get user input
query = st.text_area("Job Description or Query")

# Fix for the embedding model - Use the correct API based on version 0.8.4
def get_embedding(text):
    try:
        # Current Google Generative AI API (v0.8.4) uses this method for embeddings
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response["embedding"])
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        raise

# Recommend
if st.button("Recommend"):
    if not query:
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Analyzing using Gemini..."):
            try:
                query_vec = get_embedding(query)
                
                # Create a progress bar for embedding generation
                progress_bar = st.progress(0)
                total_items = len(df)
                
                # Create empty score column
                df["score"] = 0.0
                
                # Process in batches to show progress
                for i, row in enumerate(df.iterrows()):
                    index, data = row
                    try:
                        doc_vec = get_embedding(data["full_text"])
                        df.at[index, "score"] = cosine_similarity([query_vec], [doc_vec])[0][0]
                    except Exception as e:
                        # Use the correct column name 'name' instead of 'Assessment Name'
                        assessment_name = data.get("name", "Unknown")
                        st.warning(f"Error processing assessment {assessment_name}: {e}")
                    
                    # Update progress
                    progress = int((i + 1) / total_items * 100)
                    progress_bar.progress(progress)
                
                # Sort and display results
                top_df = df.sort_values("score", ascending=False).head(10)
                st.success("Top Recommended Assessments:")
                
                # Create a copy for display formatting
                display_df = top_df.copy()
                
                # Format the URL column with markdown links
                display_df["url"] = display_df.apply(
                    lambda row: f"[Link]({row['url']})", axis=1
                )
                
                # Rename and format other columns
                display_df = display_df.rename(columns={
                    "name": "Assessment Name",
                    "url": "URL",  
                    "duration": "Duration",
                    "remote_testing": "Remote Testing Support",
                    "adaptive_irt": "Adaptive/IRT Support",
                    "test_type": "Test Type",
                    "score": "Relevance Score"
                })
                
                # Format score as percentage
                display_df["Relevance Score"] = display_df["Relevance Score"].apply(lambda x: f"{x:.2%}")
                
                # Display with markdown enabled for URLs
                st.dataframe(
                    display_df,
                    column_config={
                        "URL": st.column_config.LinkColumn()
                    },
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error during recommendation: {e}")
