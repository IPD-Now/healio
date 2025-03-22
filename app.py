from fastapi import FastAPI
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash"  # Updated model supporting grounding

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
            genai.Content(
                role="user",
                parts=[genai.Part.from_text(f"User: {msg[0]}\nDr. Healio: {msg[1]}")]
            )
            for msg in history
        ]

        # Add the latest user input
        conversation_history.append(
            genai.Content(
                role="user",
                parts=[genai.Part.from_text(user_input)]
            )
        )

        # Define tools (Google Search Grounding)
        tools = [genai.Tool(google_search=genai.GoogleSearch())]

        # Generate response with Google Search Grounding
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            contents=conversation_history,
            tools=tools,
            stream=False  # Set to `True` if you want streaming responses
        )

        # Extract response text
        if response and response.text:
            response_text = response.text.strip()
            history.append((user_input, response_text))
        else:
            response_text = "Sorry, I couldn't fetch a response."

        return {"response": response_text, "history": history}

    except Exception as e:
        return {"error": str(e)}

@app.post("/reset_chat")
def reset_chat():
    return {"history": []}
