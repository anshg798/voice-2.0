import os
import uuid
import tempfile
import whisper
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from gtts import gTTS

# --- FastAPI App ---
app = FastAPI(title="Voice Translation App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Whisper model (small recommended for better Hindi recognition) ---
model = whisper.load_model("small")

# --- Temp dir ---
TEMP_DIR = tempfile.gettempdir()


@app.post("/translate")
async def translate_audio(file: UploadFile, target_lang: str = Form(...), force_lang: str = Form(None)):
    """
    Upload audio + target language.
    Optional: force_lang="hi" (or any ISO code) to guide transcription.
    """
    try:
        # Save uploaded file
        suffix = os.path.splitext(file.filename)[-1] or ".mp3"
        temp_audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}{suffix}")
        with open(temp_audio_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        # --- Speech-to-Text ---
        whisper_args = {"fp16": False}
        if force_lang:
            whisper_args["language"] = force_lang

        result = model.transcribe(temp_audio_path, **whisper_args)
        original_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        # If transcription is garbage for Hindi, retry with language="hi"
        if (detected_lang not in ["hi", "en"]) or len(original_text) < 3:
            retry_result = model.transcribe(temp_audio_path, fp16=False, language="hi")
            original_text = retry_result["text"].strip()
            detected_lang = retry_result.get("language", "hi")

        # --- Translation ---
        translated_text = GoogleTranslator(source="auto", target=target_lang).translate(original_text)

        # --- TTS ---
        tts_filename = f"{uuid.uuid4().hex}.mp3"
        tts_path = os.path.join(TEMP_DIR, tts_filename)
        gTTS(translated_text, lang=target_lang).save(tts_path)

        # Clean up uploaded file
        try:
            os.remove(temp_audio_path)
        except:
            pass

        # Return **relative URL**
        audio_url = f"/tts/{tts_filename}"

        return {
            "detected_language": detected_lang,
            "original_text": original_text,
            "translated_text": translated_text,
            "target_language": target_lang,
            "audio_url": audio_url
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/tts/{filename}")
async def get_tts(filename: str):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(
            filepath,
            media_type="audio/mpeg",
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    return JSONResponse({"error": "File not found"}, status_code=404)
