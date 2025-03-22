from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import requests

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

MODEL_NAME = "models/gemini-2.0-flash"

# Fetch Dr. Healio Prompt from external source
PROMPT_URL = "https://gist.githubusercontent.com/shudveta/3286f04b7bc36a94bb9b84065fdc64a0/raw/075cd1541681d8fc8efb78cb9750d75f58c1af70/prompt.txt"
try:
    response = requests.get(PROMPT_URL)
    response.raise_for_status()
    DR_HEALIO_PROMPT = response.text.strip()
except requests.exceptions.RequestException:
    DR_HEALIO_PROMPT = "Dr. Healio Default Prompt"

# Define request structure
class ChatRequest(BaseModel):
    user_input: str
    history: list = []  # History will be sent in the request

@app.get("/")
def read_root():
    return {"message": "Dr. Healio API is running!"}

@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history  # Received history from the request

        # Construct full history
        full_history = DR_HEALIO_PROMPT + "\n" + "\n".join(
            [f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in history]
        )

        # Generate AI response with Google Search grounding
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            full_history + f"\nUser: {user_input}\nDr. Healio:",
            generation_config={"mode": "google_search_retrieval"}  # ✅ Properly enabling search
        )

        # Append new interaction
        history.append((user_input, response.text))

        return {"response": response.text, "history": history}
    except Exception as e:
        return {"error": str(e)}


@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
