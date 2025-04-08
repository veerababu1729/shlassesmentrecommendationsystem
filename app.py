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
            lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}. This test is designed to assess {row['test_type']} skills.", 
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

# Check if an assessment is relevant based on keywords in query
def is_relevant_to_query(assessment_text, query_terms):
    assessment_text = assessment_text.lower()
    # Extract important words from query (remove common words)
    query_terms = [term.lower() for term in query_terms.split() 
                  if len(term) > 2 and term.lower() not in ['the', 'and', 'for', 'with']]
    
    # Count how many query terms appear in the assessment text
    matches = sum(1 for term in query_terms if term in assessment_text)
    
    # Consider relevant if at least one term matches or if the query is very short
    return matches > 0 or len(query_terms) == 0

# Function to filter relevant assessments by excluding obvious mismatches
def filter_irrelevant_tests(df, query):
    query = query.lower()
    
    # Special case handling for common language/tech queries
    filtered_df = df.copy()
    
    # Handle programming language specific queries
    if 'java ' in query or query == 'java' or 'java developer' in query:
        # For Java queries, prioritize Java tests and exclude non-relevant languages
        relevant_terms = ['java', 'javascript', 'sql', 'programming', 'coding', 'developer']
        filtered_df = filtered_df[filtered_df['full_text'].str.lower().apply(
            lambda text: any(term in text.lower() for term in relevant_terms))]
        
    elif 'python' in query:
        # For Python queries, prioritize Python tests
        relevant_terms = ['python', 'programming', 'data science', 'sql', 'coding']
        filtered_df = filtered_df[filtered_df['full_text'].str.lower().apply(
            lambda text: any(term in text.lower() for term in relevant_terms))]
    
    # Add more language/tech specific filters as needed
    
    # If filtering removed all options, revert to original dataset
    if len(filtered_df) == 0:
        return df
    
    return filtered_df

# Dynamic threshold based on overall score distribution
def get_dynamic_threshold(scores, default_min=0.60):
    if len(scores) < 5:
        return default_min
    
    # Use statistical approaches to find natural cutoffs
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # More aggressive threshold: mean + 0.25*std deviation 
    threshold = mean_score + 0.25 * std_score
    
    # Don't go below our minimum acceptable threshold
    return max(threshold, default_min)

# Recommend
if st.button("Recommend"):
    if not query:
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Finding relevant assessments..."):
            try:
                # First filter obviously irrelevant tests based on keyword matching
                filtered_df = filter_irrelevant_tests(df, query)
                
                query_vec = get_embedding(query)
                
                # Create a progress bar for embedding generation
                progress_bar = st.progress(0)
                total_items = len(filtered_df)
                
                # Create empty score column
                filtered_df["score"] = 0.0
                
                # Process in batches to show progress
                for i, row in enumerate(filtered_df.iterrows()):
                    index, data = row
                    try:
                        doc_vec = get_embedding(data["full_text"])
                        similarity_score = cosine_similarity([query_vec], [doc_vec])[0][0]
                        
                        # Give a boost to assessments that match keywords in query
                        if is_relevant_to_query(data["full_text"], query):
                            similarity_score += 0.1  # Boost matching assessments
                        
                        filtered_df.at[index, "score"] = similarity_score
                    except Exception as e:
                        assessment_name = data.get("name", "Unknown")
                        st.warning(f"Error processing assessment {assessment_name}: {e}")
                    
                    # Update progress
                    progress = int((i + 1) / total_items * 100)
                    progress_bar.progress(progress)
                
                # Sort by relevance score
                sorted_df = filtered_df.sort_values("score", ascending=False)
                
                # Get a dynamic threshold based on score distribution
                all_scores = sorted_df["score"].values
                dynamic_threshold = get_dynamic_threshold(all_scores)
                
                # Filter by relevance threshold to ensure results are relevant
                relevant_df = sorted_df[sorted_df["score"] >= dynamic_threshold]
                
                # Ensure we show between 1-10 results
                if len(relevant_df) == 0:
                    top_df = sorted_df.head(1)
                    st.warning("No highly relevant assessments found. Showing best match.")
                elif len(relevant_df) > 10:
                    top_df = relevant_df.head(10)
                else:
                    top_df = relevant_df
                
                st.success(f"Found {len(top_df)} Recommended Assessments:")
                
                # Create a copy for display formatting
                display_df = top_df.copy()
                
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
                
                # Display with proper link column configuration
                st.dataframe(
                    display_df[["Assessment Name", "URL", "Test Type", "Duration", "Relevance Score"]],
                    column_config={
                        "URL": st.column_config.LinkColumn("Link")
                    },
                    hide_index=True
                )
                
            except Exception as e:
                st.error(f"Error during recommendation: {e}")
