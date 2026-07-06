import fitz
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Dual-Analysis Precision ATS API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 🔒 قراءة المفتاح بشكل آمن من بيئة الاستضافة
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")

client = genai.Client(api_key=GEMINI_API_KEY)

class SkillMatchElement(BaseModel):
    required_skill: str = Field(description="Technical skill extracted from the Job Description.")
    is_matched: bool = Field(description="True if this skill (or a semantic equivalent) exists in the candidate's extracted CV skills.")
    matching_evidence: Optional[str] = Field(description="The matching skill or proof found in the CV. Leave empty if False.")

class DualATSStrictSchema(BaseModel):
    all_candidate_cv_skills: List[str] = Field(description="Comprehensive list of ALL technical skills, tools, languages, and frameworks discovered independently in the CV.")
    all_job_description_skills: List[str] = Field(description="Comprehensive list of ALL technical skills, tools, languages, and frameworks required independently in the Job Description.")
    skills_cross_matching: List[SkillMatchElement] = Field(description="The systematic cross-reference mapping between Job skills and CV skills.")

def extract_text_from_pdf(pdf_bytes):
    try:
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + " "
        return text.strip()
    except Exception:
        return ""

@app.post("/match")
async def match(cv: UploadFile = File(...), job_description: str = Form(...)):
    pdf_bytes = await cv.read()
    cv_text = extract_text_from_pdf(pdf_bytes)

    if len(cv_text) < 30:
        return {
            "match_score": 0.0,
            "result": "Unreadable CV",
            "matched_skills": [],
            "missing_skills": ["Please ensure your CV is a text-based PDF."]
        }

    prompt = f"""
    Perform a rigorous dual-document analysis. You must analyze the Candidate's CV and the Job Description independently before performing cross-matching.
    
    Follow this execution pipeline:
    Step 1: Read the Candidate's CV intently. Extract EVERY single technical skill, tool, framework, library, and programming language into 'all_candidate_cv_skills'.
    Step 2: Read the Job Description intently. Extract EVERY core technical skill and tool requested into 'all_job_description_skills'.
    Step 3: Cross-reference the two extracted lists. For each skill in 'all_job_description_skills', check if it is satisfied by the skills in 'all_candidate_cv_skills' (allow semantic equivalents like 'Bloc' satisfying 'State Management'). If matched, provide the explicit evidence.
    
    Candidate CV Text:
    '''{cv_text}'''
    
    Job Description Text:
    '''{job_description}'''
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DualATSStrictSchema,
                temperature=0.0,  
                system_instruction="You are a meticulous technical auditor. You extract raw data from both sources independently, then match them based strictly on professional tech stacks."
            ),
        )

        result_json = json.loads(response.text)
        cross_matching = result_json.get("skills_cross_matching", [])
        
        if not cross_matching:
            raise HTTPException(status_code=400, detail="Could not perform cross-matching on the provided texts.")

        matched_skills = []
        missing_skills = []

        for item in cross_matching:
            skill_name = item.get("required_skill")
            if item.get("is_matched") and item.get("matching_evidence"):
                matched_skills.append(skill_name)
            else:
                missing_skills.append(skill_name)

        total_reqs = len(cross_matching)
        matched_count = len(matched_skills)
        
        score = (matched_count / total_reqs) * 100 if total_reqs > 0 else 0.0
        score = round(score, 1)

        if score >= 80: eval_result = "Excellent Match"
        elif score >= 60: eval_result = "Good Match"
        elif score >= 40: eval_result = "Average Match"
        else: eval_result = "Weak Match"

        return {
            "match_score": score,
            "result": eval_result,
            "matched_skills": list(set(matched_skills)),
            "missing_skills": list(set(missing_skills)),
            "debug_cv_skills": result_json.get("all_candidate_cv_skills", []),
            "debug_jd_skills": result_json.get("all_job_description_skills", [])
        }

    except Exception as e:
        print(f"Dual-Analysis ATS Error: {e}")
        raise HTTPException(status_code=500, detail="Error executing dual-document analysis.")

if __name__ == "__main__":
    # ⚙️ إعداد البورت ليتوافق مع Render
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Running Dual-Analysis Precision ATS Server on http://0.0.0.0:{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)