📄 SHL Assessment Recommendation System – Solution Documentation
Author: [Your Name]
Live UI Webapp: https://shlassessmentrecommendation.streamlit.app
API Endpoint: https://shlassessementapiendpoint.onrender.com/recommend
GitHub Repo: [Add your GitHub repo link here]

✅ Objective
Create a system to recommend the most relevant SHL assessments based on a job description or query. Deliver both:

STEP 1: An interactive web application.

STEP 2: A REST API endpoint that returns JSON recommendations.

🧠 Approach Summary
STEP 1 – Web UI (Streamlit)
Built using Streamlit for a clean user interface.

Users input a job description or role requirement.

The system embeds both query and test descriptions using Google Gemini Embedding API (models/embedding-001).

Computes cosine similarity to recommend the top 10 relevant SHL assessments.

Displays results in a sortable, clickable table with URLs to the test pages.

STEP 2 – JSON API Endpoint (Render)
Created a FastAPI-based endpoint /recommend.

Accepts a POST request with a query string.

Returns a ranked list of the top 5 assessments in JSON.

Deployed the backend using Render.com for public accessibility.

🛠️ Technologies Used
Component	Tool / Library
Embedding Model	Google Gemini embedding-001
Web App	Streamlit
REST API	FastAPI
Deployment	Streamlit Cloud, Render
Data Processing	pandas, NumPy, scikit-learn
Hosting	GitHub (for version control)
📦 API Usage
POST /recommend
URL: https://shlassessementapiendpoint.onrender.com/recommend

Request:

json
Copy
Edit
{
  "query": "Looking for cognitive ability and reasoning assessments"
}
Response:

json
Copy
Edit
{
  "recommendations": [
    {
      "name": "Logical Reasoning Test",
      "test_type": "Cognitive",
      "duration": "30 mins",
      "url": "https://example.com/logical-test"
    },
    ...
  ]
}
