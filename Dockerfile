# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install build dependencies for pyaudio
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
