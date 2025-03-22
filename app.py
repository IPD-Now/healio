from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from google.genai import types
import os
import requests

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

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
    history: list = []  # Chat history

@app.get("/")
def read_root():
    return {"message": "Dr. Healio API is running!"}

@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history  # Received chat history

        # Construct full conversation history
        full_history = DR_HEALIO_PROMPT + "\n" + "\n".join(
            [f"User: {msg[0]}\nDr. Healio: {msg[1]}" for msg in history]
        )

        # Create content structure
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_history + f"\nUser: {user_input}\nDr. Healio:")]
            )
        ]

        # Enable Google Search grounding
        tools = [types.Tool(google_search=types.GoogleSearch())]

        # Configure generation settings
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=tools,
            response_mime_type="text/plain",
        )

        # Generate AI response with search grounding
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )

        # Extract response text
        ai_response = response.text if response.text else "Sorry, I couldn't find relevant information."

        # Append new interaction
        history.append((user_input, ai_response))

        return {"response": ai_response, "history": history}
    except Exception as e:
        return {"error": str(e)}

@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
