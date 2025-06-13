import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Streamlit page settings
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# Custom dark theme styling
st.markdown(
    """
    <style>
        body { background-color: #121212; color: #ffffff; }
        .stApp { background-color: #1e1e1e; padding: 20px; border-radius: 10px; }
        .stMarkdown { font-size: 18px; }
        .stTextArea textarea { background-color: #333; color: #ffffff; }
        .stFileUploader div { background-color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar instructions
st.sidebar.title("📌 Instructions")
st.sidebar.info(
    "### Step 1: Enter the Job Description\n"
    "Provide a detailed job description in the input box.\n\n"
    "### Step 2: Upload Resumes (PDF)\n"
    "Upload multiple resumes in PDF format.\n\n"
    "### Step 3: AI Processing\n"
    "The system will analyze and rank candidates based on relevance.\n\n"
    "### Step 4: View Results\n"
    "Check the ranked resumes in a sorted table."
)

# PDF text extraction function
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

# Resume ranking function (returns percentage)
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], vectors[1:]).flatten() * 100  # Convert to percentage

# App Title
st.markdown("<h1 style='text-align: center; color: #ff9800;'>📄 AI Resume Screener</h1>", unsafe_allow_html=True)

# Job Description input
st.subheader("📝 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File Uploader
st.subheader("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Main logic: ranking and display
if uploaded_files and job_description:
    st.subheader("📊 Ranking Resumes")
    progress_bar = st.progress(0)
    status_text = st.empty()

    resumes = [extract_text_from_pdf(file) for file in uploaded_files]

    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f"Processing {file.name}...")

    time.sleep(1)
    progress_bar.empty()
    status_text.success("Processing Completed ✅")

    if any(resumes):
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Score (%)": scores
        })

        # Sort by highest match
        results = results.sort_values(by="Score (%)", ascending=False)

        # Display results
        st.dataframe(
            results.style
                .format({"Score (%)": "{:.2f}"})
                .bar(subset=["Score (%)"], color="#ff9800")
        )

        st.balloons()
        st.success("✅ Ranking Completed! Check the Table Above.")
    else:
        st.warning("⚠️ No text could be extracted from the uploaded resumes.")
