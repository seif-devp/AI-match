import re
from fastapi import FastAPI, File, UploadFile, Form
import PyPDF2
import io
import uvicorn
import spacy
import spacy.cli

app = FastAPI()

# تحميل الموديل الذكي والخفيف من spaCy (حجمه 12 ميجا بس وبيفهم الجرامر)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# قائمة كلمات الحشو الإدارية اللي مش عايزينها تطلع في المتطلبات
TRASH_WORDS = [
    "team", "experience", "development", "company", "requirements", "skills", 
    "years", "working", "software", "understanding", "work", "knowledge", "ability",
    "role", "environment", "benefits", "salary", "projects", "candidate", "opportunity"
]

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
    text = text.replace(" rn ", " react native ")
    text = text.replace("rest apis", "api")
    return text

def extract_smart_keywords(text: str) -> list:
    doc = nlp(text)
    keywords = set()
    
    # استخراج "الكتل الاسمية" لأن المهارات دايماً بتكون أسماء مش أفعال
    for chunk in doc.noun_chunks:
        # بنشيل أدوات التعريف زي a, an, the عشان الكلمة تطلع نظيفة
        clean_chunk = " ".join([token.text for token in chunk if token.pos_ != "DET"])
        clean_chunk = clean_chunk.lower().strip()
        
        # بنفلتر الكلمات الإدارية، وبنأكد إن المصطلح ميزيدش عن 3 كلمات
        if clean_chunk and clean_chunk not in TRASH_WORDS and len(clean_chunk.split()) <= 3:
            keywords.add(clean_chunk)
            
    return list(keywords)

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

    # 1. استخراج الكلمات الذكية من الوظيفة (JD)
    jd_keywords = extract_smart_keywords(clean_jd)

    # 2. البحث عن هذه الكلمات في السيرة الذاتية (CV)
    # المرة دي هندور على المهارات المطلوبة جوه الـ CV عشان منضيعش ولا كلمة
    matched_keywords = []
    cv_text_lower = clean_cv.lower()

    for kw in jd_keywords:
        if re.search(rf'\b{re.escape(kw)}\b', cv_text_lower):
            matched_keywords.append(kw)

    # 3. حساب النسبة
    if len(jd_keywords) == 0:
        ai_score = 0
    else:
        ai_score = (len(matched_keywords) / len(jd_keywords)) * 100

    return {
        "status": "success",
        "ai_score": round(ai_score),
        "version": "spacy_nlp_v1", # ده الإصدار الجديد الخفيف والذكي
        "keywords": jd_keywords,
        "matched_keywords": matched_keywords
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)