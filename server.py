from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import tempfile
import time  # New import for timing
from dotenv import load_dotenv

# 1. Load environment variables AT THE VERY TOP
load_dotenv()

# Double check API Key existence for debugging
if not os.environ.get("GROQ_API_KEY"):
    print("❌ WARNING: GROQ_API_KEY not found in .env!")
else:
    print("✅ GROQ_API_KEY loaded successfully.")

from src.audio.transcriber import AudioTranscriber
from src.privacy.redactor import PIIRedactor
from src.intelligence.analyzer import LLMAnalyzer

app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🔥 Initialize fast cloud services
print("🚀 Initializing VoiceOps Sentinel Intelligence Pipeline (Cloud)...")
transcriber = AudioTranscriber()
redactor = PIIRedactor()
analyzer = LLMAnalyzer()

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("static/index.html")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), enable_diarization: bool = True):
    start_time = time.time()
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        audio_path = tmp_file.name

    try:
        # Run Pipeline with Timers
        print(f"\n--- Processing: {file.filename} (Diarization: {enable_diarization}) ---")
        
        # A. Transcription & Diarization
        t1 = time.time()
        # Update transcriber.transcribe to accept the flag
        diarized_text = transcriber.transcribe(audio_path, run_diarization=enable_diarization)
        t2 = time.time()
        print(f"⏱️ Step 1 (Transcription/Diarization): {t2 - t1:.2f}s")
        
        # B. Redaction
        t3 = time.time()
        redacted_text = redactor.redact(diarized_text)
        t4 = time.time()
        print(f"⏱️ Step 2 (PII Redaction): {t4 - t3:.2f}s")
        
        # C. Intelligence Analysis
        t5 = time.time()
        analysis_raw = analyzer.analyze(redacted_text)
        t6 = time.time()
        print(f"⏱️ Step 3 (Intelligence Analysis): {t6 - t5:.2f}s")
        
        print(f"🚀 TOTAL PIPELINE TIME: {t6 - start_time:.2f}s")
        
        # In case analyzer returns a string, try to parse
        import json
        try:
            analysis = json.loads(analysis_raw)
        except:
            analysis = {"Summary": analysis_raw, "Sentiment": "Unknown"}

        return {
            "diarized_text": diarized_text,
            "redacted_text": redacted_text,
            "analysis": analysis
        }
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
