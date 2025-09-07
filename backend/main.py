from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import os
import tempfile
from pathlib import Path
import mlx_whisper
import mlx_lm
from mlx_lm import load, generate
import asyncio
from asyncio import Queue

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="YouTube Video Summarizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str

class SummaryResponse(BaseModel):
    title: str
    summary: str
    transcript: str

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model = None
tokenizer = None

# Queue for processing requests one at a time
processing_queue = Queue(maxsize=10)
queue_worker_started = False

async def queue_worker():
    """Worker to process summarization requests one at a time"""
    while True:
        try:
            # Get the next request from the queue
            request_data = await processing_queue.get()
            if request_data is None:  # Shutdown signal
                break
            
            request, future = request_data
            
            try:
                # Process the request
                result = await process_summarization_request(request)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                processing_queue.task_done()
                
        except Exception as e:
            print(f"Error in queue worker: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Load model and start queue worker on application startup"""
    global model, tokenizer, queue_worker_started
    
    try:
        print(f"Loading model {MODEL_NAME}...")
        model, tokenizer = load(MODEL_NAME)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            MODEL_NAME_FALLBACK = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            print(f"Trying fallback model {MODEL_NAME_FALLBACK}...")
            model, tokenizer = load(MODEL_NAME_FALLBACK)
            print("Fallback model loaded successfully!")
        except Exception as e2:
            print(f"Error with fallback model: {e2}")
            model, tokenizer = None, None
    
    # Start the queue worker
    if not queue_worker_started:
        asyncio.create_task(queue_worker())
        queue_worker_started = True
        print("Queue worker started")

def extract_audio_from_youtube(url: str) -> tuple[str, str]:
    """Extraire l'audio d'une vidéo YouTube et retourner le chemin du fichier audio et le titre"""
    
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Titre non disponible')
            
            ydl.download([url])
            
            audio_file = None
            for file in os.listdir(temp_dir):
                if file.startswith("audio") and file.endswith(".wav"):
                    audio_file = os.path.join(temp_dir, file)
                    break
            
            if not audio_file or not os.path.exists(audio_file):
                raise Exception("Audio file not found after extraction")
                
            return audio_file, title
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during audio extraction: {str(e)}")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using MLX Whisper"""
    try:
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
        return result["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

def generate_summary(transcript: str, title: str) -> str:
    """Generate a blog post summary from transcript using MLX-LM"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Generation model not available")
    
    try:
        # Multilingual prompt for generating blog post summary
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert blog writer. Create an engaging and well-structured blog article from a YouTube video transcription. Always write the summary in the SAME language as the original video transcription.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Video Title: {title}

Transcription: {transcript[:4000]}

Create a well-structured blog article with:
1. An engaging introduction
2. Key points organized in clear sections
3. A conclusion
4. Use a professional but accessible tone
5. Add subheadings to structure the content

IMPORTANT: Write the entire summary in the same language as the transcription. Do not translate to English.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=1000,
        )
        
        generated_text = response.strip()
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            summary = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            summary = generated_text
            
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summary generation: {str(e)}")

async def process_summarization_request(request: VideoRequest) -> SummaryResponse:
    """Process a single summarization request"""
    try:
        print(f"Extracting audio from: {request.url}")
        audio_path, title = extract_audio_from_youtube(request.url)
        
        print("Transcribing audio...")
        transcript = transcribe_audio(audio_path)
        
        print("Generating summary...")
        summary = generate_summary(transcript, title)
        
        # Clean up temporary files
        try:
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))
        except:
            pass
        
        return SummaryResponse(
            title=title,
            summary=summary,
            transcript=transcript
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_video(request: VideoRequest):
    """Main endpoint to summarize a YouTube video using queue system"""
    
    # Check if queue is full
    if processing_queue.full():
        raise HTTPException(status_code=429, detail="Server is busy. Please try again later.")
    
    # Create a future to get the result
    future = asyncio.Future()
    
    # Add request to queue
    try:
        await processing_queue.put((request, future))
        print(f"Request queued for: {request.url}")
        
        # Wait for the result
        result = await future
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "YouTube Video Summarizer API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)