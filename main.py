import os
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemini-2.0-flash"

# External prompt URL
PROMPT_URL = "https://gist.githubusercontent.com/shudveta/3286f04b7bc36a94bb9b84065fdc64a0/raw/075cd1541681d8fc8efb78cb9750d75f58c1af70/prompt.txt"

# Fetch Dr. Healio Prompt
try:
    response = requests.get(PROMPT_URL)
    response.raise_for_status()
    DR_HEALIO_PROMPT = response.text.strip()
except requests.exceptions.RequestException as e:
    DR_HEALIO_PROMPT = "Dr. Healio Default Prompt: Unable to load external prompt."
    print(f"Error fetching prompt: {e}")

# FastAPI App
app = FastAPI()

# Chat History
chat_history = []

# Request Model
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def dr_healio_chat(request: ChatRequest):
    global chat_history
    try:
        # Create conversation history
        full_history = DR_HEALIO_PROMPT + "\n".join([f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in chat_history])

        # Generate response
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(full_history + f"\nUser: {request.user_input}\nDr. Healio:")

        # Update history
        chat_history.append((request.user_input, response.text))

        return {"response": response.text, "history": chat_history}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "history": chat_history}

@app.post("/reset")
async def reset_chat():
    global chat_history
    chat_history = []
    return {"response": "Chat reset successfully"}
