# File: handler.py
# RunPod handler for Coqui TTS (XTTS) with voice cloning
# Handles HTTP requests for text-to-speech generation with voice cloning from reference audio

import os
import io
import logging
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
_model_loaded = False
_tts = None


def _load_model() -> bool:
    """Load the XTTS model."""
    global _model_loaded, _tts
    
    if _model_loaded:
        logger.info("Model already loaded")
        return True
    
    logger.info("Loading Coqui XTTS model...")
    try:
        from TTS.api import TTS
        
        # XTTS v2 model with voice cloning support
        # This model supports multilingual voice cloning via speaker_wav
        import torch
        _tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        if torch.cuda.is_available():
            _tts.to("cuda")
        _model_loaded = True
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def _download_file(url: str, dest_dir: Path) -> Optional[Path]:
    """Download a file from URL to destination directory."""
    import requests
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Determine filename from URL or generate UUID
        filename = url.split("/")[-1].split("?")[0]
        if not filename or "." not in filename:
            filename = f"reference_{uuid.uuid4().hex[:8]}.wav"
        
        dest_path = dest_dir / filename
        dest_path.write_bytes(response.content)
        logger.info(f"Downloaded file to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def _synthesize_audio(
    text: str,
    reference_audio_path: Optional[str] = None,
    language: str = "ru",
    **kwargs
) -> tuple[bytes, Optional[str]]:
    """
    Synthesize audio using XTTS with voice cloning.
    
    Args:
        text: Text to synthesize.
        reference_audio_path: Path to reference WAV file for voice cloning.
        language: Language code (default: "ru").
        
    Returns:
        Tuple of (audio_bytes, error_message).
    """
    global _tts
    
    try:
        import numpy as np
        import soundfile as sf
        
        logger.info(f"Synthesizing: text={text[:50]}..., lang={language}, ref={reference_audio_path}")
        
        # Generate audio
        # XTTS voice cloning via speaker_wav (string path, per README example)
        wav = _tts.tts(
            text=text,
            speaker_wav=reference_audio_path,
            language=language,
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, 24000, format="WAV")
        audio_bytes = buffer.getvalue()
        
        logger.info(f"Generated audio: {len(audio_bytes)} bytes")
        return audio_bytes, None
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None, str(e)


def handler(event, context=None):
    """
    Main RunPod handler function.
    
    RunPod passes the "input" object from the POST /run request.
    
    Args:
        event: Dict containing request parameters (from event["input"])
            - text: str - Text to synthesize (required)
            - reference_audio_url: str - URL of reference audio for voice cloning (required)
            - language: str - Language code (default: "ru")
            - storage: dict - S3 credentials for upload (optional)
                - endpoint: str
                - bucket: str
                - access_key: str
                - secret_key: str
            - temperature: float - (ignored by XTTS, kept for compatibility)
            - exaggeration: float - (ignored by XTTS, kept for compatibility)
            - cfg_weight: float - (ignored by XTTS, kept for compatibility)
            - seed: int - (ignored by XTTS, kept for compatibility)
                
        context: RunPod context (unused)
    
    Returns:
        Dict with response:
            - If storage provided: {"status": "completed", "output": {"audio": "<base64>"}}
            - On error: {"status": "failed", "error": "..."}
    """
    global _model_loaded
    
    # RunPod passes input object directly
    data = event.get("input", event)
    
    # Load model if not loaded
    if not _model_loaded:
        if not _load_model():
            return {
                "status": "failed",
                "error": "Failed to load TTS model"
            }
    
    # Extract parameters from data
    text = data.get("text", "")
    reference_audio_url = data.get("reference_audio_url")
    storage = data.get("storage")
    language = data.get("language", "ru")
    
    # Validate required params
    if not text:
        return {
            "status": "failed",
            "error": "Missing required parameter: text"
        }
    
    if not reference_audio_url:
        return {
            "status": "failed",
            "error": "Missing required parameter: reference_audio_url"
        }
    
    logger.info(f"Processing TTS request: text={text[:50]}..., lang={language}")
    
    # Create temp directory for reference audio
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        reference_audio_path = None
        
        # Download reference audio
        logger.info(f"Downloading reference audio from {reference_audio_url}")
        reference_audio_path = _download_file(reference_audio_url, temp_path)
        if not reference_audio_path:
            return {
                "status": "failed",
                "error": f"Failed to download reference audio from {reference_audio_url}"
            }
        
        # Synthesize audio
        audio_bytes, error = _synthesize_audio(
            text=text,
            reference_audio_path=str(reference_audio_path),
            language=language,
        )
        
        if error:
            return {
                "status": "FAILED",
                "output": {"error": error, "audio_base64": None}
            }
        
        # Handle output based on storage
        if storage:
            # Upload to S3
            audio_url = _upload_to_s3(audio_bytes, storage)
            if not audio_url:
                return {
                    "status": "FAILED",
                    "output": {"error": "Failed to upload audio to S3", "audio_base64": None}
                }
            # For S3, return the URL in a compatible format
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return {
                "status": "COMPLETED",
                "output": {"audio_base64": audio_b64, "audio_url": audio_url, "error": None}
            }
        else:
            # Return base64 encoded audio
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return {
                "status": "COMPLETED",
                "output": {"audio_base64": audio_b64, "error": None}
            }


def _upload_to_s3(audio_bytes: bytes, storage: dict) -> Optional[str]:
    """Upload audio bytes to S3 and return URL."""
    import boto3
    
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=storage.get("endpoint"),
            aws_access_key_id=storage.get("access_key"),
            aws_secret_access_key=storage.get("secret_key"),
        )
        
        key = f"tts_audio/{uuid.uuid4().hex}.wav"
        s3_client.put_object(
            Bucket=storage.get("bucket"),
            Key=key,
            Body=audio_bytes,
            ContentType="audio/wav",
        )
        
        # Build public URL
        endpoint = storage.get("endpoint", "").rstrip("/")
        bucket = storage.get("bucket")
        url = f"{endpoint}/{bucket}/{key}"
        
        logger.info(f"Uploaded to S3: {url}")
        return url
        
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


# --- For local testing ---
if __name__ == "__main__":
    # Test handler locally
    test_event = {
        "text": "Привет, это тест!",
        "reference_audio_url": "https://s3.firstvds.ru/celebrity-videos/voice_samples/test.wav",
        "language": "ru",
    }
    
    if _load_model():
        result = handler(test_event)
        print(result)
