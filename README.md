# YouTube Video Summarizer

A web application that generates intelligent summaries of YouTube videos using AI.

The application uses React for the frontend, FastAPI for the backend, MLX Whisper for audio transcription, and MLX-LM with the Llama-3.2-3B-Instruct-4bit model for summary generation.

## Prerequisites

- Python 3.11+
- Node.js 22+
- macOS with MLX support (for AI)

## Installation and Usage:

1. Clone the repository
2. Start the server
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Start the client
   ```
   cd frontend
   npm install
   npm run dev
   ```
4. Open the application in your browser
   ```
   http://localhost:5173
   ```
5. Paste the YouTube video URL in the input field
6. Click on the "Summarize" button
7. Wait for the summary to be generated (the process may take a few minutes)
8. Read the generated summary

## Features

- ✅ Automatic audio extraction from YouTube videos
- ✅ Audio transcription to text with MLX Whisper
- ✅ Intelligent summary generation with MLX-LM
- ✅ Modern and responsive user interface
- ✅ Real-time progress tracking
- ✅ Multilingual support (automatic language detection)
- ✅ Asynchronous processing for better performance

## Architecture

- **Frontend**: React + TypeScript + Vite + shadcn/ui
- **Backend**: FastAPI + Python
- **AI**: MLX Whisper (transcription) + MLX-LM (summaries)
- **Video extraction**: yt-dlp
