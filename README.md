# Youtube Video Summarizer

A simple web application that summarizes youtube videos using AI.

The application is built using React, FastAPI, MLX Whisper and MLX-LM (using Llama-3.2-3B-Instruct-4bit for summarization).

Usage :

1. Clone the repository
2. Start the server
   ```
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
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
5. Paste the youtube video URL in the input field
6. Click on the "Summarize" button
7. Wait for the summary to be generated
8. Read the summary
