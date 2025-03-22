from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from google.genai import types
import os
import requests

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"

PROMPT_URL = "https://gist.githubusercontent.com/shudveta/3286f04b7bc36a94bb9b84065fdc64a0/raw/075cd1541681d8fc8efb78cb9750d75f58c1af70/prompt.txt"
try:
    response = requests.get(PROMPT_URL)
    response.raise_for_status()
    DR_HEALIO_PROMPT = response.text.strip()
except requests.exceptions.RequestException:
    DR_HEALIO_PROMPT = "Dr. Healio Default Prompt"

class ChatRequest(BaseModel):
    user_input: str
    history: list = []

@app.get("/")
def read_root():
    return {"message": "Dr. Healio API is running!"}

@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history

        full_history = DR_HEALIO_PROMPT + "\n" + "\n".join(
            [f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in history]
        )

        model = genai.GenerativeModel(MODEL_NAME)

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_history + f"\nUser: {user_input}\nDr. Healio:")],
            ),
        ]

        tools = [types.Tool(google_search=types.GoogleSearch())]

        generate_content_config = types.GenerateContentConfig(tools=tools)

        response = model.generate_content(contents=contents, config=generate_content_config)

        history.append((user_input, response.text))

        return {"response": response.text, "history": history}
    except Exception as e:
        return {"error": str(e)}

@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
