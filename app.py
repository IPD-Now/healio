from fastapi import FastAPI
from pydantic import BaseModel
import os
import google.generativeai as genai
from google.generativeai import types

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

MODEL_NAME = "models/gemini-2.0-flash"

# Define request structure
class ChatRequest(BaseModel):
    user_input: str
    history: list = []  # History will be sent in the request

@app.get("/")
def read_root():
    return {"message": "Dr. Healio API with Google Search Grounding is running!"}

@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history  # Received history from the request

        # Construct conversation history
        conversation_history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"User: {msg[0]}\nDr. Healio: {msg[1]}")]
            )
            for msg in history
        ]

        # Add the latest user input
        conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )

        # Define tools (Google Search Grounding)
        tools = [types.Tool(google_search=types.GoogleSearch())]

        # Generate response with Google Search Grounding
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=tools,
            response_mime_type="text/plain",
        )

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            contents=conversation_history,
            config=generate_content_config,
        )

        # Append new interaction
        if response and response.candidates:
            response_text = response.candidates[0].content.parts[0].text.strip()
            history.append((user_input, response_text))
        else:
            response_text = "Sorry, I couldn't fetch a response."

        return {"response": response_text, "history": history}

    except Exception as e:
        return {"error": str(e)}

@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
