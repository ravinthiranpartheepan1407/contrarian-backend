from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import PyPDF2
import io
import os
import json
import ssl
import urllib.request

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contrarian.vercel.app", "https://contrarian-ventures-psn3s8uj2.vercel.app/"],  # Only allow the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResult(BaseModel):
    team_score: dict
    business_model_score: dict
    traction_score: dict
    total_score: float
    geography: str
    industry: str
    stage: str
    rationale: dict


def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def get_phi3_response(prompt, api_key):
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,  # Set to 0 for consistent analysis
        "top_p": 1,
        "stream": False
    }

    body = str.encode(json.dumps(data))
    url = os.getenv("API_URL")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result['choices'][0]['message']['content']
    except urllib.error.HTTPError as error:
        raise Exception(f"API request failed with status code: {error.code}\n{error.read().decode('utf8', 'ignore')}")


def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def analyze_with_phi3(text: str) -> dict:
    # Enable self-signed HTTPS
    allowSelfSignedHttps(True)

    # Your Phi-3 API key should be stored in environment variables
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise Exception("PHI3_API_KEY environment variable not set")

    prompt = """
    Analyze this pitch deck and provide scores based on the following criteria.
    For each criterion, assign:
    1 point for YES
    0.5 points for PARTIAL YES
    0 points for NO

    Team:
    1. Do founders have relevant experience in the field?
    2. Have the founders previously worked together?
    3. Have one or more of the founders built a business before?

    Business Model:
    1. Is the business easily scalable?
    2. Can the business add new product lines, services, upsell the customer?
    3. Is the business model immune to external shocks?

    Traction:
    1. Does the business have initial customers?
    2. Does the business exhibit rapid growth?
    3. Is there indication of good customer retention?

    Also identify:
    - Geography (where the company is based)
    - Industry/sector
    - Stage (pre-seed, seed, Series A)
    
    Summary:
    - Write summary of this startup

    Provide detailed rationale for each score.

    Return the analysis in valid JSON format exactly matching this structure:
    {
        "team_score": {
            "relevant_experience": {"score": float, "rationale": "string"},
            "worked_together": {"score": float, "rationale": "string"},
            "previous_business": {"score": float, "rationale": "string"}
        },
        "business_model_score": {
            "scalability": {"score": float, "rationale": "string"},
            "product_expansion": {"score": float, "rationale": "string"},
            "shock_resistance": {"score": float, "rationale": "string"}
        },
        "traction_score": {
            "initial_customers": {"score": float, "rationale": "string"},
            "growth": {"score": float, "rationale": "string"},
            "retention": {"score": float, "rationale": "string"}
        },
        "geography": "string",
        "industry": "string",
        "stage": "string",
        "summary": "string"
    }

    Pitch deck text: """ + text

    try:
        response = get_phi3_response(prompt, api_key)
        return json.loads(response)
    except json.JSONDecodeError:
        raise Exception("Failed to parse Phi-3 response as JSON")


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_pitch_deck(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)

    # Analyze with Phi-3
    analysis = analyze_with_phi3(text)

    # Calculate total score
    total_score = sum(
        [item["score"] for score_category in
         [analysis["team_score"], analysis["business_model_score"], analysis["traction_score"]]
         for item in score_category.values()]
    )

    return {
        **analysis,
        "total_score": total_score,
        "rationale": {
            "team": analysis["team_score"],
            "business_model": analysis["business_model_score"],
            "traction": analysis["traction_score"]
        }
    }