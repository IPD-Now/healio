from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from google import genai  # ✅ Using google-genai instead of google.generativeai

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)  # ✅ New Client Initialization

MODEL_NAME = "gemini-2.0-flash"

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
        full_history = [
            genai.types.Content(role="user", parts=[genai.types.Part.from_text(DR_HEALIO_PROMPT)])
        ] + [
            genai.types.Content(role="user", parts=[genai.types.Part.from_text(msg[0])]) for msg in history
        ] + [
            genai.types.Content(role="model", parts=[genai.types.Part.from_text(msg[1])]) for msg in history
        ]

        # Append user input
        full_history.append(genai.types.Content(role="user", parts=[genai.types.Part.from_text(user_input)]))

        # Generate AI response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_history
        )

        # Append new interaction
        history.append((user_input, response.candidates[0].content.parts[0].text))

        return {"response": response.candidates[0].content.parts[0].text, "history": history}
    except Exception as e:
        return {"error": str(e)}

@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
