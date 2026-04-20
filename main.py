import re
import yake
from fastapi import FastAPI, File, UploadFile, Form
import PyPDF2
import io
import uvicorn

app = FastAPI()

# إعداد YAKE لاستخراج الكلمات المفتاحية
# n=2 (بيستخرج كلمات مفردة ومركبة)، top=25 (أهم 25 كلمة)
kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=25, features=None)

# قائمة بسيطة لتنظيف النتائج من كلمات الحشو الإدارية اللي YAKE ممكن يلقطها
TRASH_WORDS = ["team", "experience", "development", "company", "requirements", "skills", "years", "working", "software"]

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
    # بنسيب الحروف الكابيتال زي ما هي عشان YAKE بيعتمد عليها في استخراج المهارات (زي MVC, RN)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # توحيد بعض الاختصارات يدوياً قبل الاستخراج
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

    # 1. استخراج الكلمات من الـ Job Description (Extraction 1)
    jd_extracted = kw_extractor.extract_keywords(clean_jd)
    jd_keywords = set([kw[0].lower() for kw in jd_extracted if kw[0].lower() not in TRASH_WORDS])

    # 2. استخراج الكلمات من الـ CV (Extraction 2)
    cv_extracted = kw_extractor.extract_keywords(clean_cv)
    cv_keywords = set([kw[0].lower() for kw in cv_extracted])

    # 3. المقارنة لاستخراج الكلمات المشتركة (Intersection)
    matched_keywords = list(jd_keywords.intersection(cv_keywords))

    # 4. حساب النسبة
    if len(jd_keywords) == 0:
        ai_score = 0
    else:
        ai_score = (len(matched_keywords) / len(jd_keywords)) * 100

    return {
        "status": "success",
        "ai_score": round(ai_score),
        "version": "yake_v1", # السطر ده عشان نتأكد
        "keywords": list(jd_keywords),       # دي هتتعرض كـ Requirements
        "matched_keywords": matched_keywords # دي هتنور بالأخضر في الـ UI
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)