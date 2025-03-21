import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemini-2.0-flash"

# Fetch Dr. Healio Prompt
PROMPT_URL = "https://gist.githubusercontent.com/shudveta/3286f04b7bc36a94bb9b84065fdc64a0/raw/075cd1541681d8fc8efb78cb9750d75f58c1af70/prompt.txt"
try:
    response = requests.get(PROMPT_URL)
    response.raise_for_status()
    DR_HEALIO_PROMPT = response.text.strip()
except requests.exceptions.RequestException:
    DR_HEALIO_PROMPT = "Dr. Healio Default Prompt: Unable to load external prompt."

# Initialize FastAPI
app = FastAPI()

# Chat request model
class ChatRequest(BaseModel):
    user_input: str
    history: list

# API Endpoint for Chat
@app.post("/dr_healio_chat")
def dr_healio_chat(data: ChatRequest):
    try:
        full_history = DR_HEALIO_PROMPT + "\n".join(
            [f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in data.history]
        )
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(full_history + f"\nUser: {data.user_input}\nDr. Healio:")

        new_history = data.history + [(data.user_input, response.text)]
        return {"history": new_history, "response": response.text}

    except Exception as e:
        return {"error": str(e)}

# API Endpoint for Reset Chat
@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
