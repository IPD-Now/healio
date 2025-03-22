from fastapi import FastAPI
from pydantic import BaseModel
import os
from google import genai
from google.genai import types

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-pro"  # Ensure you use a model that supports Google Search Grounding

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
                parts=[types.Part.from_text(f"User: {msg[0]}\nDr. Healio: {msg[1]}")]
            )
            for msg in history
        ]

        # Add latest user input
        conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(user_input)]
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

        # Call the Gemini model with Google Search
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=conversation_history,
            config=generate_content_config,
        )

        # Extract response text
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
