import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import json
import re

# Setup Streamlit page
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("🔍 SHL Assessment Recommendation System")
st.write("Paste a job description or query below:")

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

# Load data
try:
    df = pd.read_csv("data/assessments.csv")
    
    # Add descriptions for each assessment (from first code)
    descriptions = {
        "Python Programming Test": "Multi-choice test that measures the knowledge of Python programming, databases, modules and library. For Mid-Professional developers.",
        "Java Programming Test": "Multi-choice test that measures the knowledge of Java programming, databases, frameworks and libraries. For Entry-Level developers.",
        "SQL Test": "Assessment that measures SQL querying and database knowledge. For data professionals and developers.",
        "JavaScript Test": "Assessment that evaluates JavaScript programming skills including DOM manipulation and frameworks.",
        "Verify Numerical Reasoning Test": "Assessment that measures numerical reasoning ability for workplace performance.",
        "Verify Verbal Reasoning Test": "Assessment that measures verbal reasoning ability for workplace performance.",
        "Verify Coding Pro": "Advanced coding assessment for professional developers across multiple languages.",
        "OPQ Personality Assessment": "Comprehensive workplace personality assessment for job fit and development.",
        "Workplace Personality Assessment": "Assessment that evaluates workplace behavior and personality traits.",
        "Business Simulation": "Interactive business scenario simulation for evaluating decision-making skills.",
        "General Ability Test": "Assessment that measures general mental ability across various cognitive domains.",
        "Teamwork Assessment": "The Technology Job Focused Assessment assesses key behavioral attributes required for success in fast-paced, rapidly changing technology work environments."
    }
    
    # Map descriptions to dataframe (add with safe handling if column doesn't exist)
    df["description"] = df["name"].map(lambda name: descriptions.get(name, ""))
    
    # Create full_text column like in original
    if "full_text" not in df.columns:
        df["full_text"] = df.apply(
            lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}. {row.get('description', '')}", 
            axis=1
        )
    
    # Define test type mappings for API response format (from first code)
    st.session_state.test_type_mappings = {
        "Cognitive": ["Knowledge & Skills"],
        "Technical": ["Knowledge & Skills"],
        "Personality": ["Personality & Behaviour"],
        "Behavioral": ["Competencies", "Personality & Behaviour"],
        "Simulation": ["Competencies"]
    }
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Function to use Gemini to match assessments with query (from first code)
def match_assessments_with_gemini(query, assessments_data):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Create a context with all assessment information
        assessments_context = []
        for _, row in assessments_data.iterrows():
            assessment_info = {
                "name": row["name"],
                "description": row.get("description", ""),
                "test_type": row["test_type"],
                "duration": row["duration"]
            }
            assessments_context.append(assessment_info)
        
        # Create the prompt for Gemini
        prompt = f"""
        As an HR assessment recommendation system, I need to find the most relevant assessments for the following job description or query:

        Query: "{query}"

        Available assessments:
        {json.dumps(assessments_context, indent=2)}

        For this query, which assessments from the list would be most relevant? Consider:
        1. Technical skills mentioned in the query
        2. Soft skills or personality traits mentioned
        3. The specific job role or industry
        4. Required experience level if mentioned

        Identify only the assessment names that are truly relevant to the query. Do not include any assessment that doesn't directly relate to the skills or attributes mentioned in the query. For example, if the query mentions "Java developer", don't include Python assessments.

        Respond with a JSON object having this exact format:
        {{
          "relevant_assessments": ["Assessment Name 1", "Assessment Name 2", ...],
          "relevance_scores": [0.95, 0.87, ...] 
        }}
        
        The relevance_scores should be numbers between 0 and 1 indicating how relevant each assessment is.

        Your response should only include the JSON object, nothing else.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse the JSON response
        try:
            # Find JSON content between curly braces
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                assessments = result.get("relevant_assessments", [])
                scores = result.get("relevance_scores", [1.0] * len(assessments))  # Default score if missing
                
                # Return tuple of assessments and scores
                return assessments, scores
            return [], []
        except json.JSONDecodeError as e:
            st.error(f"Error parsing Gemini response: {e}")
            st.write(f"Response was: {response_text}")
            return [], []
    except Exception as e:
        st.error(f"Error in Gemini matching: {e}")
        return [], []

# Get user input
query = st.text_area("Job Description or Query")

# Recommend
if st.button("Recommend"):
    if not query:
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Finding relevant assessments..."):
            try:
                # Use Gemini to get relevant assessment names
                relevant_assessment_names, relevance_scores = match_assessments_with_gemini(query, df)
                
                # Create a dictionary mapping assessment names to scores
                score_dict = dict(zip(relevant_assessment_names, relevance_scores))
                
                # Filter the DataFrame to include only relevant assessments
                if relevant_assessment_names:
                    relevant_df = df[df["name"].isin(relevant_assessment_names)].copy()
                    # Add scores to dataframe
                    relevant_df["score"] = relevant_df["name"].map(lambda name: score_dict.get(name, 0.0))
                    # Sort by relevance score
                    relevant_df = relevant_df.sort_values("score", ascending=False)
                else:
                    # Fallback: If no matches, return a general assessment
                    st.warning("No highly relevant assessments found. Showing best match.")
                    relevant_df = df[df["name"] == "General Ability Test"].copy()
                    if relevant_df.empty:
                        relevant_df = df.head(1).copy()  # Absolute fallback
                    relevant_df["score"] = 0.6  # Default score
                
                # Ensure we have between 1-10 results
                if len(relevant_df) > 10:
                    top_df = relevant_df.head(10)
                else:
                    top_df = relevant_df
                
                st.success(f"Found {len(top_df)} Recommended Assessments:")
                
                # Create a copy for display formatting
                display_df = top_df.copy()
                
                # Format the assessment data for display
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
                
                # Additional info section (similar to first code's response format)
                with st.expander("Detailed Assessment Information"):
                    for _, row in top_df.iterrows():
                        duration_value = int(''.join(filter(str.isdigit, row["duration"])))
                        test_type_list = st.session_state.test_type_mappings.get(row["test_type"], [row["test_type"]])
                        
                        st.subheader(row["name"])
                        st.markdown(f"**Description:** {row.get('description', '')}")
                        st.markdown(f"**Duration:** {duration_value} minutes")
                        st.markdown(f"**Test Type:** {', '.join(test_type_list)}")
                        st.markdown(f"**Remote Testing:** {'Yes' if row.get('remote_testing') == 'Yes' else 'No'}")
                        st.markdown(f"**Adaptive Support:** {'Yes' if row.get('adaptive_irt') == 'Yes' else 'No'}")
                        st.markdown(f"**URL:** {row['url']}")
                        st.markdown("---")
                
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
                st.exception(e)
