import re
import yake
from fastapi import FastAPI, File, UploadFile, Form
import PyPDF2
import io
import uvicorn

app = FastAPI()

# إعداد YAKE لاستخراج الكلمات المفتاحية
kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=20, features=None)

# زودنا كلمات الحشو اللي طلعت المرة اللي فاتت عشان الموديل ينظفها أوتوماتيك
TRASH_WORDS = [
    "team", "experience", "development", "company", "requirements", "skills", 
    "years", "working", "software", "good understanding", "solid understanding", 
    "benefits flexible", "required soft", "mobile software", "engineer requirements", 
    "required technical", "software engineering", "work independently", 
    "mobile engineer", "mobile development", "requirements required", 
    "software engineer", "mobile experiences", "team required", "experienced mobile"
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
    # شيلنا السطر اللي بيمسح الفواصل والنقط! هنسيب النص بطبيعته عشان YAKE يفهم الجمل.
    # هنكتفي بتوحيد الاختصارات بس
    text = text.replace(" RN ", " React Native ")
    text = text.replace("rest apis", "api")
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

    # 1. استخراج الكلمات من الـ Job Description
    jd_extracted = kw_extractor.extract_keywords(clean_jd)
    jd_keywords = set([kw[0].lower() for kw in jd_extracted if kw[0].lower() not in TRASH_WORDS])

    # 2. البحث الذكي جوه الـ CV
    # هندور على متطلبات الوظيفة جوه نص السيرة الذاتية مباشرة عشان منضيعش ولا كلمة
    matched_keywords = []
    cv_text_lower = clean_cv.lower()

    for kw in jd_keywords:
        # استخدام Regex للبحث عن الكلمة كاملة (عشان ميتلخبطش بين dart و darting مثلاً)
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
        "version": "yake_v2", # غيرنا النسخة عشان نتأكد إن التحديث سمع
        "keywords": list(jd_keywords),
        "matched_keywords": matched_keywords
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)