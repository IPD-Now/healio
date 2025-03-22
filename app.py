from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import google.genai as genai  # ✅ Correct Import for google-genai

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.GenerativeModel("gemini-2.0-flash", api_key=API_KEY)  # ✅ Corrected Client Initialization

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
    history: list = []  # Chat history from the user

@app.get("/")
def read_root():
    return {"message": "Dr. Healio API is running!"}

@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history

        # Construct conversation history correctly
        full_history = [
            genai.types.Content(role="user", parts=[genai.types.Part.from_text(DR_HEALIO_PROMPT)])
        ]
        
        for msg in history:
            full_history.append(genai.types.Content(role="user", parts=[genai.types.Part.from_text(msg[0])]))  # User's previous input
            full_history.append(genai.types.Content(role="model", parts=[genai.types.Part.from_text(msg[1])]))  # AI's response

        # Append current user input
        full_history.append(genai.types.Content(role="user", parts=[genai.types.Part.from_text(user_input)]))

        # Generate AI response
        response = client.generate_content(contents=full_history)
        ai_response = response.candidates[0].content.parts[0].text

        # Append new interaction to history
        history.append((user_input, ai_response))

        return {"response": ai_response, "history": history}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
