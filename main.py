from fastapi import FastAPI, File, UploadFile, Form
from sklearn.feature_extraction.text import CountVectorizer
import PyPDF2
import io
import uvicorn
import re

app = FastAPI()

# 1. قاموس المهارات التقنية (Whitelist)
# الموديل مش هيشوف أي كلمة بره القاموس ده. تقدر تزود فيه أي مهارات تانية.
TECH_SKILLS_VOCAB = [
    "flutter", "dart", "ios", "android", "native", "react native", "swift", "kotlin",
    "sap", "commerce cloud", "analytics", "ga4", "mixpanel", "firebase",
    "devops", "ci cd", "github actions", "bitbucket", "codemagic",
    "state management", "widgets", "architecture", "design patterns", "mvvm",
    "bff", "api", "rest", "backend", "security", "performance tuning",
    "agile", "scrum", "ui", "ux", "sqlite", "caching", "notifications"
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
    # إزالة الرموز وتحويل النص لحروف صغيرة
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()

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
    
    # 2. إجبار الموديل على استخدام قاموس المهارات فقط
    # ngram_range=(1, 2) عشان يقرا المهارات المكونة من كلمتين زي "state management"
    vectorizer = CountVectorizer(vocabulary=TECH_SKILLS_VOCAB, ngram_range=(1, 2))
    
    try:
        # استخراج المهارات الموجودة في الـ Job Description فقط
        jd_matrix = vectorizer.fit_transform([clean_jd])
        jd_vector = jd_matrix.toarray().flatten()
        
        # تجميع المهارات اللي ظهرت في الـ JD
        jd_keywords = [TECH_SKILLS_VOCAB[i] for i in range(len(TECH_SKILLS_VOCAB)) if jd_vector[i] > 0]
        
    except ValueError:
        return {"status": "error", "message": "Error processing Job Description."}

    # 3. البحث عن هذه المهارات داخل الـ CV
    matched_keywords = []
    for kw in jd_keywords:
        # البحث عن المهارة ككلمة مستقلة
        if re.search(rf'\b{re.escape(kw)}\b', clean_cv):
            matched_keywords.append(kw)

    # 4. حساب نسبة التطابق
    if len(jd_keywords) == 0:
        ai_score = 0
    else:
        ai_score = (len(matched_keywords) / len(jd_keywords)) * 100

    return {
        "status": "success",
        "ai_score": round(ai_score),
        "keywords": jd_keywords,           # دي المتطلبات التقنية الصافية للوظيفة
        "matched_keywords": matched_keywords # دي المهارات اللي المتقدم يعرفها
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)