"""
production_server.py — High-Concurrency Multi-Worker Piper TTS Server
Designed for RunPod GPU Pods (20+ concurrent conversational voice streams).

Run with Gunicorn (4 workers):
    gunicorn production_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 --timeout 60
"""

import asyncio
import io
import logging
import os
import sys
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Piper imports
from piper import PiperVoice, SynthesisConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(name)s: %(message)s",
)
logger = logging.getLogger("piper-production")

# Configuration via environment variables
DEFAULT_VOICES = [
    "hi_IN-rohan-medium",
    "en_US-lessac-medium",
    "ml_IN-meera-medium",
    "te_IN-padmavathi-medium",
]
VOICE_MODELS_ENV = os.getenv("VOICE_MODELS", ",".join(DEFAULT_VOICES))
DATA_DIR = Path(os.getenv("DATA_DIR", str(REPO_ROOT)))
USE_CUDA = os.getenv("USE_CUDA", "true").lower() in ("true", "1")

# In-memory voice storage per worker process
loaded_voices: Dict[str, PiperVoice] = {}

# Shorthand alias mappings for voice requests
VOICE_ALIASES: Dict[str, str] = {
    "hi": "hi_IN-rohan-medium",
    "hindi": "hi_IN-rohan-medium",
    "hi-in": "hi_IN-rohan-medium",
    "rohan": "hi_IN-rohan-medium",
    "en": "en_US-lessac-medium",
    "english": "en_US-lessac-medium",
    "en-us": "en_US-lessac-medium",
    "lessac": "en_US-lessac-medium",
    "ml": "ml_IN-meera-medium",
    "malayalam": "ml_IN-meera-medium",
    "ml-in": "ml_IN-meera-medium",
    "meera": "ml_IN-meera-medium",
    "te": "te_IN-padmavathi-medium",
    "telugu": "te_IN-padmavathi-medium",
    "te-in": "te_IN-padmavathi-medium",
    "padmavathi": "te_IN-padmavathi-medium",
}


def find_model_file(model_name: str, search_dir: Path) -> Optional[Path]:
    """Look for model file with or without .onnx suffix in search directories."""
    candidates = [
        search_dir / f"{model_name}.onnx",
        search_dir / model_name,
        REPO_ROOT / f"{model_name}.onnx",
        REPO_ROOT / model_name,
        Path(f"/data/{model_name}.onnx"),
        Path(f"/workspace/{model_name}.onnx"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load voice models into GPU/CPU memory on worker startup."""
    global loaded_voices
    pid = os.getpid()
    logger.info(f"Worker PID {pid} initializing voices (CUDA={USE_CUDA})...")

    models_to_load = [m.strip() for m in VOICE_MODELS_ENV.split(",") if m.strip()]
    for model_name in models_to_load:
        model_path = find_model_file(model_name, DATA_DIR)
        if model_path and model_path.exists():
            try:
                logger.info(f"Worker PID {pid} loading model '{model_name}' from {model_path}...")
                voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                clean_name = model_path.stem
                loaded_voices[clean_name] = voice
                logger.info(f"Worker PID {pid} loaded '{clean_name}' successfully")
            except Exception as e:
                logger.error(f"Worker PID {pid} failed to load voice '{model_name}': {e}")
        else:
            logger.warning(
                f"Worker PID {pid}: Model file for '{model_name}' not found in {DATA_DIR}. "
                f"Ensure models are downloaded before starting."
            )

    if not loaded_voices:
        logger.warning(f"Worker PID {pid}: No voices loaded! Waiting for dynamic load or /synthesize.")

    yield

    logger.info(f"Worker PID {pid} shutting down...")
    loaded_voices.clear()


app = FastAPI(
    title="Piper TTS Production High-Concurrency Server",
    description="Optimized multi-worker Piper TTS API for LiveKit conversational voice agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web clients / dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    voice: Optional[str] = Field(None, description="Voice name or alias (e.g. 'hi_IN-rohan-medium', 'hi', 'en', 'ml', 'te')")
    speaker_id: Optional[int] = Field(None, description="Speaker ID for multi-speaker models")
    length_scale: Optional[float] = Field(1.0, description="Speaking rate/speed (lower = faster, 1.0 = normal)")
    noise_scale: Optional[float] = Field(0.667, description="Voice variability noise")
    noise_w_scale: Optional[float] = Field(0.8, description="Phoneme width noise")


def resolve_voice(voice_query: Optional[str]) -> PiperVoice:
    """Resolve a voice instance from query name, alias, or default."""
    if not loaded_voices:
        raise HTTPException(status_code=500, detail="No voice models are currently loaded on this worker.")

    if not voice_query:
        # Default to first loaded voice
        return next(iter(loaded_voices.values()))

    v_lower = voice_query.lower().strip()
    target_id = VOICE_ALIASES.get(v_lower, voice_query)

    # 1. Direct match
    if target_id in loaded_voices:
        return loaded_voices[target_id]

    # 2. Case-insensitive or partial prefix match
    for name, voice in loaded_voices.items():
        if name.lower() == v_lower or name.lower().startswith(v_lower):
            return voice

    # Fallback to default
    logger.warning(f"Voice '{voice_query}' not found. Falling back to default '{next(iter(loaded_voices.keys()))}'")
    return next(iter(loaded_voices.values()))


def _synthesize_sync(voice: PiperVoice, text: str, syn_config: SynthesisConfig) -> bytes:
    """Synchronous synthesis executed inside thread pool."""
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize_wav(
                text=text,
                wav_file=wav_file,
                syn_config=syn_config,
                set_wav_format=True,
            )
        return wav_io.getvalue()


@app.get("/health")
def health():
    """Health check endpoint for container monitors and load balancers."""
    return {
        "status": "ok",
        "pid": os.getpid(),
        "cuda": USE_CUDA,
        "loaded_voices": list(loaded_voices.keys()),
        "voice_count": len(loaded_voices),
    }


@app.get("/voices")
def list_voices():
    """List loaded voices and their audio parameters."""
    voices_info = {}
    for name, v in loaded_voices.items():
        voices_info[name] = {
            "sample_rate": v.config.sample_rate,
            "num_speakers": v.config.num_speakers,
            "espeak_voice": v.config.espeak_voice,
        }
    return {"voices": voices_info, "aliases": VOICE_ALIASES}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """
    Synthesize audio from text with sub-100ms response time.
    Returns 16-bit PCM WAV audio.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    voice = resolve_voice(req.voice)

    syn_config = SynthesisConfig(
        speaker_id=req.speaker_id if req.speaker_id is not None else voice.config.default_speaker_id,
        length_scale=float(req.length_scale if req.length_scale is not None else voice.config.length_scale),
        noise_scale=float(req.noise_scale if req.noise_scale is not None else voice.config.noise_scale),
        noise_w_scale=float(req.noise_w_scale if req.noise_w_scale is not None else voice.config.noise_w_scale),
    )

    try:
        # Offload blocking ONNX synthesis to thread so async event loop stays responsive
        wav_bytes = await asyncio.to_thread(_synthesize_sync, voice, text, syn_config)
    except Exception as e:
        logger.error(f"Synthesis failed for text '{text[:40]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    # Direct execution for quick local debug (single worker)
    uvicorn.run("production_server:app", host="0.0.0.0", port=5000, reload=False)
