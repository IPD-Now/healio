from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import requests
from google.genai import types

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")

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
    history: list = []  
    use_search: bool = False  # NEW: Enables Google Search Grounding

@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history if request.history else []

        # Construct full history
        full_history = DR_HEALIO_PROMPT + "\n" + "\n".join(
            [f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in history]
        )

        client = genai.Client(api_key=API_KEY)

        # Enable Google Search Grounding if use_search=True
        tools = [types.Tool(google_search=types.GoogleSearch())] if request.use_search else []

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=tools,
            response_mime_type="text/plain",
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_history + f"\nUser: {user_input}\nDr. Healio:")]
                )
            ],
            config=generate_content_config,
        )

        # Append new interaction
        reply = response.text if hasattr(response, "text") else "No response generated."
        history.append([user_input, reply])

        return {"response": reply, "history": history}
    
    except Exception as e:
        return {"error": str(e)}

