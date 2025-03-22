@app.post("/dr_healio_chat")
def dr_healio_chat(request: ChatRequest):
    try:
        user_input = request.user_input
        history = request.history  # Received history from the request

        # Construct full history correctly
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
