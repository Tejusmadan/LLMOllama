import os
import spacy
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

nlp = spacy.load("en_core_web_sm")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

class EntityModel(BaseModel):
    text: str
    label: str

class ProcessResponse(BaseModel):
    entities: List[EntityModel]
    llm_response: str

@app.post("/process", response_model=ProcessResponse)
def process_prompt(req: PromptRequest):
    prompt_text = req.prompt.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    doc = nlp(prompt_text)
    entities: List[Dict[str, str]] = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    print("\n[spaCy NER] Detected named entities:")
    if entities:
        for ent in entities:
            print(f"  • {ent['text']}  ({ent['label']})")
    else:
        print("  • None")

    payload = {
        "model": "llama2",  
        "prompt": prompt_text,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result_json = resp.json()
        llm_text = result_json.get("response", "").strip()
    except Exception as e:
        print(f"[Ollama API] Error calling /api/generate: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from LLM.")

    print("\n[LLM Response]")
    print(llm_text)
    print("---------------------------------------------------\n")

    return {"entities": entities, "llm_response": llm_text}
