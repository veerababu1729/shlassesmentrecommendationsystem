import streamlit as st
import pandas as pd
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
    
    # List available models (helpful for debugging)
    available_models = [m.name for m in genai.list_models()]
    gemini_models = [m for m in available_models if 'gemini' in m.lower()]
    
    # Debug expander for developers
    with st.expander("Debug Info (Available Models)"):
        st.write("Available Gemini Models:")
        st.write(gemini_models)
    
    # Choose the right model - try newer naming conventions if available
    model_name = None
    
    # Look for gemini models with priority order
    preferred_models = ['gemini-1.5-pro', 'gemini-pro', 'gemini-1.0-pro']
    for preferred in preferred_models:
        matches = [m for m in gemini_models if preferred in m]
        if matches:
            model_name = matches[0]
            break
    
    # If no match found, use the first available gemini model
    if not model_name and gemini_models:
        model_name = gemini_models[0]
    
    # If still no model found, show error
    if not model_name:
        st.error("No Gemini models available with your API key. Please check your API key permissions.")
        st.stop()
        
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
    
    # Map descriptions to dataframe
    df["description"] = df["name"].map(descriptions)
    
    # Define test type mappings for API response format
    test_type_mappings = {
        "Cognitive": ["Knowledge & Skills"],
        "Technical": ["Knowledge & Skills"],
        "Personality": ["Personality & Behaviour"],
        "Behavioral": ["Competencies", "Personality & Behaviour"],
        "Simulation": ["Competencies"]
    }
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Function to use Gemini to match assessments with query
def match_assessments_with_gemini(query, assessments_data):
    try:
        # Use the model name we determined earlier
        model = genai.GenerativeModel(model_name)
        
        # Create a context with all assessment information
        assessments_context = []
        for _, row in assessments_data.iterrows():
            assessment_info = {
                "name": row["name"],
                "description": row["description"],
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
          "relevant_assessments": ["Assessment Name 1", "Assessment Name 2", ...]
        }}

        Your response should only include the JSON object, nothing else.
        """
        
        # Use safety settings if necessary
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Set generation config
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Generate content with updated parameters
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response_text = response.text.strip()
        
        # Parse the JSON response
        try:
            # Find JSON content between curly braces
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result.get("relevant_assessments", [])
            
            # If the regex didn't match, try to parse the entire response as JSON
            try:
                result = json.loads(response_text)
                return result.get("relevant_assessments", [])
            except:
                st.error("Failed to parse Gemini JSON response")
                st.code(response_text)
                return []
                
        except json.JSONDecodeError as e:
            st.error(f"Error parsing Gemini response: {e}")
            st.write(f"Response was: {response_text}")
            return []
    except Exception as e:
        st.error(f"Error in Gemini matching: {e}")
        st.exception(e)
        return []

# Get user input
query = st.text_area("Job Description or Query")

# Recommend
if st.button("Recommend"):
    if not query:
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Finding relevant assessments..."):
            try:
                # Get relevant assessment names using Gemini
                relevant_assessment_names = match_assessments_with_gemini(query, df)
                
                # Filter the DataFrame to include only relevant assessments
                if relevant_assessment_names:
                    relevant_df = df[df["name"].isin(relevant_assessment_names)]
                else:
                    # Fallback: If no matches, return a general assessment
                    st.warning("No highly relevant assessments found. Showing best match.")
                    relevant_df = df[df["name"] == "General Ability Test"]
                    if relevant_df.empty:
                        relevant_df = df.head(1)  # Absolute fallback
                
                # Format the assessment data for the response
                recommended_assessments = []
                for _, row in relevant_df.iterrows():
                    duration_value = int(''.join(filter(str.isdigit, row["duration"])))
                    test_type_list = test_type_mappings.get(row["test_type"], [row["test_type"]])
                    
                    assessment = {
                        "url": row["url"],
                        "adaptive_support": "Yes" if row["adaptive_irt"] == "Yes" else "No",
                        "description": row["description"],
                        "duration": duration_value,
                        "remote_support": "Yes" if row["remote_testing"] == "Yes" else "No",
                        "test_type": test_type_list,
                        "name": row["name"]  # Adding name for display purposes
                    }
                    recommended_assessments.append(assessment)
                
                # Ensure we have at least one assessment
                if not recommended_assessments:
                    # Absolute fallback - use first assessment in dataset
                    first_row = df.iloc[0]
                    duration_value = int(''.join(filter(str.isdigit, first_row["duration"])))
                    test_type_list = test_type_mappings.get(first_row["test_type"], [first_row["test_type"]])
                    
                    fallback_assessment = {
                        "url": first_row["url"],
                        "adaptive_support": "Yes" if first_row["adaptive_irt"] == "Yes" else "No",
                        "description": first_row["description"],
                        "duration": duration_value,
                        "remote_support": "Yes" if first_row["remote_testing"] == "Yes" else "No",
                        "test_type": test_type_list,
                        "name": first_row["name"]  # Adding name for display purposes
                    }
                    recommended_assessments = [fallback_assessment]
                elif len(recommended_assessments) > 10:
                    recommended_assessments = recommended_assessments[:10]
                
                # Create DataFrame from recommendation results for display
                results_df = pd.DataFrame(recommended_assessments)
                
                # Convert test_type list to string for display
                results_df["test_type_str"] = results_df["test_type"].apply(lambda x: ", ".join(x))
                
                # Show number of results
                st.success(f"Found {len(recommended_assessments)} Recommended Assessments:")
                
                # Display table with the same headings used in API endpoint
                st.dataframe(
                    results_df[["name", "url", "description", "duration", "remote_support", "adaptive_support", "test_type_str"]].rename(columns={
                        "name": "Assessment Name",
                        "url": "URL",
                        "description": "Description",
                        "duration": "Duration (minutes)",
                        "remote_support": "Remote Support",
                        "adaptive_support": "Adaptive Support",
                        "test_type_str": "Test Type"
                    }),
                    column_config={
                        "URL": st.column_config.LinkColumn("URL")
                    },
                    hide_index=True
                )
                
            except Exception as e:
                st.error(f"Error during recommendation: {e}")
                st.exception(e)
# Footer
st.markdown("---")
st.markdown("© 2025 | Built with ❤️ using Streamlit and Gemini API")
