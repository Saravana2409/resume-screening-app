import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def rank_resumes(job_desc, resumes):
    cleaned_jd = clean_text(job_desc)
    resume_texts = [clean_text(text) for text in resumes]
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([cleaned_jd] + resume_texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    return scores

def get_common_keywords(jd_text, resume_text):
    jd_words = set(clean_text(jd_text).split())
    resume_words = set(clean_text(resume_text).split())
    return sorted(jd_words & resume_words)

# Streamlit UI
st.set_page_config(page_title="Resume Screening Tool", page_icon="📄")
st.title("📄 Resume Screening Tool (NLP-Based)")
st.markdown("Upload resumes and paste the job description to get ranked candidates!")

uploaded_files = st.file_uploader("Upload Resumes (.pdf or .docx)", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Paste Job Description Here")

if st.button("🧠 Screen Resumes"):
    if not uploaded_files or not job_description.strip():
        st.warning("Please upload resumes and paste a job description.")
    else:
        resume_names = []
        resume_texts = []

        for file in uploaded_files:
            file_type = file.name.split('.')[-1]
            resume_names.append(file.name)

            if file_type == "pdf":
                resume_texts.append(extract_text_from_pdf(file))
            elif file_type == "docx":
                resume_texts.append(extract_text_from_docx(file))
            else:
                resume_names.pop()
                st.warning(f"Unsupported file: {file.name}")
        
        scores = rank_resumes(job_description, resume_texts)
        ranked_resumes = sorted(zip(resume_names, resume_texts, scores), key=lambda x: x[2], reverse=True)

        st.subheader("📊 Ranked Resumes")
        results = []
        for idx, (name, text, score) in enumerate(ranked_resumes, start=1):
            st.markdown(f"### {idx}. 📄 **{name}**")
            st.write(f"**Match Score:** `{score*100:.2f}%`")
            st.progress(score)

            common_keywords = get_common_keywords(job_description, text)
            if common_keywords:
                st.caption(f"✅ Common keywords: `{', '.join(common_keywords)}`")
            else:
                st.caption("⚠️ No common keywords found.")
            results.append({"Resume": name, "Match Score (%)": round(score * 100, 2), "Common Keywords": ', '.join(common_keywords)})

        # Download results
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Ranked Results as CSV", csv, "ranked_resumes.csv", "text/csv")
