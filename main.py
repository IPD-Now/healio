import os
import base64
import tempfile
import wave
import asyncio
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import traceback


app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Create the genai client
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    raise ValueError(f"Failed to initialize genai client: {e}")

# Define model names
DEFAULT_MODEL = "gemini-2.0-flash"

# Fetch Dr. Healio Prompt from external source
PROMPT_URL = "https://gist.githubusercontent.com/shudveta/3286f04b7bc36a94bb9b84065fdc64a0/raw/ee7c72fd0db520e018a740e6baa57c69e8f7304a/prompt.txt"
try:
    response = requests.get(PROMPT_URL)
    response.raise_for_status()
    DR_HEALIO_PROMPT = response.text.strip()
except requests.exceptions.RequestException as e:
    print(f"Warning: Failed to fetch prompt, using default. Error: {e}")
    DR_HEALIO_PROMPT = "Dr. Healio Default Prompt"

# Define request structure
class ChatRequest(BaseModel):
    user_input: str
    history: list = []

@app.get("/")
def read_root():
    return {"message": "Dr. Healio API is running with Google Search Grounding (google-genai)!"}

@app.post("/dr_healio_chat")
async def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history

        full_history = DR_HEALIO_PROMPT + "\n" + "\n".join(
            [f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in history]
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"{full_history}\nUser: {user_input}\nDr. Healio:")],
            ),
        ]
        tools = [types.Tool(google_search=types.GoogleSearch())]
        generate_content_config = types.GenerateContentConfig(
            temperature=0.9,
            top_p=0.85,
            top_k=30,
            max_output_tokens=1500,
            tools=tools,
            response_mime_type="text/plain",
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=DEFAULT_MODEL,  # Using default model for regular chat
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text

        history.append((user_input, response_text))
        return {"response": response_text, "history": history}

    except Exception as e:
        print("Full exception trace:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/ping")
def ping():
    return {"status": "alive", "message": "Dr. Healio backend is up and running."}

@app.post("/reset_chat")
async def reset_chat():
    return {"history": []}
