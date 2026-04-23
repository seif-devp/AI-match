from fastapi import FastAPI, UploadFile, File, Form
import spacy
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="CV Matcher API (PDF Support)")

nlp = spacy.load("en_core_web_sm")


# ---------------------------
# Extract text from PDF
# ---------------------------

def extract_text_from_pdf(pdf_bytes):

    text = ""

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

        for page in doc:

            text += page.get_text()

    return text


# ---------------------------
# Clean text
# ---------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    return text


# ---------------------------
# Extract keywords
# ---------------------------

def extract_keywords(text):

    doc = nlp(text)

    keywords = []

    for token in doc:

        if token.pos_ in ["NOUN", "PROPN", "ADJ", "NUM"]:

            keywords.append(token.text)

    return list(set(keywords))


# ---------------------------
# Similarity score
# ---------------------------

def similarity_score(cv_text, job_text):

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform([cv_text, job_text])

    score = cosine_similarity(vectors[0], vectors[1])

    return float(score[0][0])


# ---------------------------
# API Endpoint
# ---------------------------

@app.post("/match-pdf")

async def match_pdf(
    cv_file: UploadFile = File(...),
    job_description: str = Form(...)
):

    pdf_bytes = await cv_file.read()

    cv_text = extract_text_from_pdf(pdf_bytes)

    cv_clean = clean_text(cv_text)

    job_clean = clean_text(job_description)

    cv_keywords = extract_keywords(cv_clean)

    job_keywords = extract_keywords(job_clean)

    score = similarity_score(cv_clean, job_clean)

    matched_keywords = list(set(cv_keywords) & set(job_keywords))

    return {

        "similarity_score": round(score * 100, 2),

        "matched_keywords": matched_keywords,

        "cv_keywords": cv_keywords,

        "job_keywords": job_keywords

    }