import os
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from leverparser import ResumeParser
import tempfile

app = FastAPI()

@app.post("/analyze")
async def analyze_cv(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    # 1. حفظ ملف الـ PDF مؤقتاً لأن ResumeParser بيحتاج مسار ملف (Path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        content = await cv_file.read()
        temp_pdf.write(content)
        temp_path = temp_pdf.name

    try:
        # 2. تشغيل الـ Parser النضيف
        parser = ResumeParser()
        resume_data = parser.parse(temp_path)

        # استخراج المهارات وسنين الخبرة من الـ CV
        cv_skills = [s.lower() for s in resume_data.skills]
        years_of_exp = resume_data.get_years_experience()

        # 3. استخراج مهارات الـ JD (هنستخدم Parser برضه للنص)
        # بما إن الـ Parser بياخد ملفات، هنكتب الـ JD في ملف مؤقت برضه
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_jd:
            temp_jd.write(jd_text.encode('utf-8'))
            jd_data = parser.parse(temp_jd.name)
            jd_skills = [s.lower() for s in jd_data.skills]

        # 4. المقارنة
        matched_keywords = list(set(jd_skills).intersection(set(cv_skills)))
        
        # حساب السكور (بناءً على المهارات المشتركة)
        if not jd_skills:
            score = 0
        else:
            score = (len(matched_keywords) / len(jd_skills)) * 100

        return {
            "status": "success",
            "ai_score": round(score),
            "version": "lever_parser_v1",
            "experience_years": years_of_exp,
            "keywords": jd_skills,
            "matched_keywords": matched_keywords
        }

    finally:
        # مسح الملفات المؤقتة عشان الرامات والديسك
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)