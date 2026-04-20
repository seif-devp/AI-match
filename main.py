from fastapi import FastAPI, File, UploadFile, Form
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import PyPDF2
import io
import uvicorn
import re

app = FastAPI()

# القاموس الأسود (عشان نمنع كلمات الحشو الإدارية من الاستخراج)
CUSTOM_STOP_WORDS = [
    "used", "using", "implemented", "developed", "worked", "created", "built", 
    "experience", "skills", "education", "university", "project", "projects", 
    "app", "application", "software", "system", "designed", "managed", "team",
    "year", "years", "month", "months", "knowledge", "good", "excellent", "high",
    "end", "hands", "ensure", "platforms", "systems", "management", "delivery",
    "development", "standards", "lead", "business", "working", "requirements", 
    "role", "practices", "technology", "technical", "understanding", "deep", 
    "proven", "strong", "ability", "responsibilities", "ideal", "candidate",
    "mobile", "apps", "performance", "architecture", "key", "seeking", "visionary",
    "related", "field", "degree", "science", "understanding", "cycle", "principles"
]
ALL_STOP_WORDS = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    # توحيد الاختصارات عشان لو الـ JD كاتب RN وإنت كاتب React Native
    text = text.replace("rn ", "react native ")
    text = text.replace("rest apis", "api")
    text = text.replace("rest api", "api")
    return text

@app.post("/analyze")
async def analyze_cv(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    pdf_content = await cv_file.read()
    raw_cv_text = extract_text_from_pdf(pdf_content)

    if not raw_cv_text.strip():
        return {"status": "error", "message": "Could not extract text from CV."}

    clean_cv = clean_text(raw_cv_text)
    clean_jd = clean_text(jd_text)
    
    try:
        # 1. Extraction from Job Description (استخراج أهم 20 مصطلح من الوظيفة)
        jd_vectorizer = TfidfVectorizer(stop_words=ALL_STOP_WORDS, ngram_range=(1, 2), max_features=20)
        jd_vectorizer.fit([clean_jd])
        jd_keywords = set(jd_vectorizer.get_feature_names_out())

        # 2. Extraction from CV (استخراج أهم 40 مصطلح من السيرة الذاتية)
        # بندي الـ CV فرصة أكبر (40 كلمة) عشان نطلع كل مهارات المتقدم
        cv_vectorizer = TfidfVectorizer(stop_words=ALL_STOP_WORDS, ngram_range=(1, 2), max_features=40)
        cv_vectorizer.fit([clean_cv])
        cv_keywords = set(cv_vectorizer.get_feature_names_out())
        
    except ValueError:
        return {"status": "error", "message": "Text is too short for extraction."}

    # 3. المقارنة (Intersection)
    # بنجيب الكلمات المشتركة بين المجموعتين
    matched_keywords = list(jd_keywords.intersection(cv_keywords))
    
    # 4. حساب نسبة التطابق بناءً على الكلمات المشتركة
    if len(jd_keywords) == 0:
        ai_score = 0
    else:
        ai_score = (len(matched_keywords) / len(jd_keywords)) * 100

    return {
        "status": "success",
        "ai_score": round(ai_score),
        "jd_keywords": list(jd_keywords),    # الكلمات المستخرجة من الوظيفة (للعرض في فلاتر)
        "cv_keywords": list(cv_keywords),    # الكلمات المستخرجة من الـ CV 
        "matched_keywords": matched_keywords # الكلمات المتطابقة (عشان تلونها أخضر في الـ UI)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)