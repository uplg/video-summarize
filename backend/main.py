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
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict

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

class TaskStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    GENERATING_SUMMARY = "generating_summary"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoRequest(BaseModel):
    url: str

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int  # 0-100
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class SummaryResponse(BaseModel):
    title: str
    summary: str
    transcript: str

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model = None
tokenizer = None

# Task storage and queue system
tasks: Dict[str, TaskStatusResponse] = {}
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
            
            task_id, request = request_data
            
            try:
                # Process the request with task tracking
                await process_summarization_request_with_tracking(task_id, request)
            except Exception as e:
                # Update task status to failed
                if task_id in tasks:
                    tasks[task_id].status = TaskStatus.FAILED
                    tasks[task_id].error = str(e)
                    tasks[task_id].updated_at = datetime.now()
                print(f"Error processing task {task_id}: {e}")
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
    """Extract audio from YouTube video and return path and title"""
    
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }, {
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp3',
        }],
        'postprocessor_args': [
            '-af', 'atempo=1.25'  # Accelerate audio by 25%
        ],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Titre non disponible')
            
            ydl.download([url])
            
            audio_file = None
            for file in os.listdir(temp_dir):
                if file.startswith("audio") and file.endswith(".mp3"):
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

Transcription: {transcript}

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

async def process_summarization_request_with_tracking(task_id: str, request: VideoRequest):
    """Process a single summarization request with progress tracking"""
    try:
        # Update task status: extracting audio
        tasks[task_id].status = TaskStatus.EXTRACTING_AUDIO
        tasks[task_id].progress = 10
        tasks[task_id].message = "Extracting audio from YouTube video..."
        tasks[task_id].updated_at = datetime.now()
        
        print(f"Extracting audio from: {request.url}")
        # Run in thread to avoid blocking the event loop
        audio_path, title = await asyncio.to_thread(extract_audio_from_youtube, request.url)
        
        # Update task status: transcribing
        tasks[task_id].status = TaskStatus.TRANSCRIBING
        tasks[task_id].progress = 40
        tasks[task_id].message = "Transcribing audio to text..."
        tasks[task_id].updated_at = datetime.now()
        
        print("Transcribing audio...")
        # Run in thread to avoid blocking the event loop
        transcript = await asyncio.to_thread(transcribe_audio, audio_path)
        
        # Update task status: generating summary
        tasks[task_id].status = TaskStatus.GENERATING_SUMMARY
        tasks[task_id].progress = 70
        tasks[task_id].message = "Generating blog post summary..."
        tasks[task_id].updated_at = datetime.now()
        
        print("Generating summary...")
        # Run in thread to avoid blocking the event loop
        summary = await asyncio.to_thread(generate_summary, transcript, title)
        
        # Clean up temporary files
        try:
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))
        except:
            pass
        
        # Update task status: completed
        tasks[task_id].status = TaskStatus.COMPLETED
        tasks[task_id].progress = 100
        tasks[task_id].message = "Summary generated successfully!"
        tasks[task_id].result = {
            "title": title,
            "summary": summary,
            "transcript": transcript
        }
        tasks[task_id].updated_at = datetime.now()
        
    except Exception as e:
        # Update task status: failed
        tasks[task_id].status = TaskStatus.FAILED
        tasks[task_id].error = str(e)
        tasks[task_id].message = f"Error: {str(e)}"
        tasks[task_id].updated_at = datetime.now()
        raise

@app.post("/summarize", response_model=TaskResponse)
async def summarize_video(request: VideoRequest):
    """Start a new summarization task and return task ID for tracking"""
    
    # Check if queue is full
    if processing_queue.full():
        raise HTTPException(status_code=429, detail="Server is busy. Please try again later.")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task entry
    now = datetime.now()
    tasks[task_id] = TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0,
        message="Task queued for processing...",
        created_at=now,
        updated_at=now
    )
    
    # Add request to queue
    try:
        await processing_queue.put((task_id, request))
        print(f"Task {task_id} queued for: {request.url}")
        
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task created successfully. Use the task_id to check progress."
        )
        
    except Exception as e:
        # Remove task from storage if queueing failed
        if task_id in tasks:
            del tasks[task_id]
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status and progress of a specific task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]

@app.get("/tasks")
async def get_all_tasks():
    """Get all tasks (for debugging purposes)"""
    return {"tasks": list(tasks.values())}

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed or failed task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot delete a task that is still processing")
    
    del tasks[task_id]
    return {"message": "Task deleted successfully"}

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