# File: handler.py
# RunPod handler for Coqui TTS (XTTS) with voice cloning
# Identical interface to Chatterbox handler for seamless replacement

import os
import io
import logging
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np
import soundfile as sf

# Fix torch.load compatibility for XTTS speaker files
import os
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["PYTORCH_WEIGHTS_ONLY"] = "0"

import torch
torch.serialization.add_safe_globals([
    "TTS.tts.models.xtts.GPT",
    "TTS.tts.models.xtts.Vocoder",
    "TTS.tts.models.xtts.XttsInference",
])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Global State ---
_model_loaded = False
_tts_model = None


def _load_model() -> bool:
    """
    Load the XTTS model if not already loaded.
    Returns True if model is loaded successfully.
    """
    global _model_loaded, _tts_model
    
    if _model_loaded:
        logger.info("Model already loaded")
        return True
    
    logger.info("Loading Coqui XTTS model...")
    try:
        from TTS.api import TTS
        
        _tts_model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        )
        
        if torch.cuda.is_available():
            _tts_model.to("cuda")
        
        _model_loaded = True
        logger.info("XTTS model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load XTTS model: {e}")
        return False


def _download_file(url: str, temp_dir: Path) -> Optional[Path]:
    """
    Download a file from URL to a temporary directory.
    """
    import requests
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        filename = None
        if 'content-disposition' in response.headers:
            import re
            match = re.search(r'filename="?([^";\n]+)"?', response.headers['content-disposition'])
            if match:
                filename = match.group(1)
        
        if not filename:
            filename = url.split('/')[-1].split('?')[0]
            if not filename or '.' not in filename:
                filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
        
        filepath = temp_dir / filename
        filepath.write_bytes(response.content)
        logger.info(f"Downloaded {url} to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def _upload_to_s3(audio_bytes: bytes, storage: Dict[str, str]) -> Optional[str]:
    """
    Upload audio bytes to S3-compatible storage.
    """
    import boto3
    from botocore.config import Config
    
    try:
        filename = f"tts_output_{uuid.uuid4().hex[:8]}.wav"
        
        s3_params = {
            'endpoint_url': storage.get('endpoint'),
            'aws_access_key_id': storage.get('access_key'),
            'aws_secret_access_key': storage.get('secret_key'),
        }
        
        if storage.get('region'):
            s3_params['region_name'] = storage.get('region')
        
        s3_params['config'] = Config(s3={'addressing_style': 'path'})
        s3_client = boto3.client('s3', **s3_params)
        
        bucket = storage.get('bucket')
        s3_client.put_object(
            Bucket=bucket,
            Key=filename,
            Body=audio_bytes,
            ContentType='audio/wav'
        )
        
        endpoint = storage.get('endpoint', '').rstrip('/')
        public_url = f"{endpoint}/{bucket}/{filename}"
        
        logger.info(f"Uploaded to S3: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return None


def _synthesize_audio(
    text: str,
    reference_audio_path: Optional[str] = None,
    language: str = "ru"
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Synthesize audio from text using XTTS with voice cloning.
    
    Args:
        text: Text to synthesize
        reference_audio_path: Path to reference audio for voice cloning
        language: Language code (default: "ru")
        
    Returns:
        Tuple of (audio_bytes, error_message)
    """
    global _tts_model
    
    try:
        logger.info(f"Synthesizing: text={text[:50]}..., lang={language}, ref={reference_audio_path}")
        
        # XTTS: speaker_wav must be a list of paths
        speaker_wav = [reference_audio_path] if reference_audio_path else None
        
        wav = _tts_model.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
        )
        
        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            audio_np = wav.cpu().numpy()
        else:
            audio_np = np.array(wav)
        
        # Handle different output shapes
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val
        
        # Create WAV in memory (XTTS outputs 24kHz)
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, 24000, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        logger.info(f"Generated audio: {len(audio_bytes)} bytes")
        return audio_bytes, None
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return None, str(e)


def handler(event, context=None):
    """
    Main RunPod handler function.
    IDENTICAL interface to Chatterbox handler.
    
    Args:
        event: Dict containing request parameters
            - text: str - Text to synthesize (required)
            - reference_audio_url: str - URL of reference audio for voice cloning (required)
            - storage: dict - S3 credentials for upload (optional)
            - temperature: float - (ignored by XTTS)
            - exaggeration: float - (ignored by XTTS)
            - cfg_weight: float - (ignored by XTTS)
            - seed: int - (ignored by XTTS)
            - language: str - Language code (default: "ru")
            
    Returns:
        Dict with response:
            - If storage provided: {"status": "success", "audio_url": "..."}
            - If no storage: {"status": "success", "audio": "<base64 encoded wav>"}
            - On error: {"status": "error", "error": "..."}
    """
    global _model_loaded
    
    # Load model if not loaded
    if not _model_loaded:
        if not _load_model():
            return {
                "status": "error",
                "error": "Failed to load TTS model"
            }
    
    # Extract parameters from event - SAME AS CHATTERBOX
    # Bot sends {"input": {...}} so we need to unwrap
    input_data = event.get("input", event)
    
    text = input_data.get("text", "")
    reference_audio_url = input_data.get("reference_audio_url")
    storage = input_data.get("storage")
    
    # Generation parameters (XTTS only uses language, others ignored for compatibility)
    temperature = input_data.get("temperature", 0.8)
    exaggeration = input_data.get("exaggeration", 0.5)
    cfg_weight = input_data.get("cfg_weight", 0.5)
    seed = input_data.get("seed", 0)
    language = input_data.get("language", "ru")
    
    # Validate required params
    if not text:
        return {
            "status": "error",
            "error": "Missing required parameter: text"
        }
    
    if not reference_audio_url:
        return {
            "status": "error",
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
                "status": "error",
                "error": f"Failed to download reference audio from {reference_audio_url}"
            }
        
        # Synthesize audio
        audio_bytes, error = _synthesize_audio(
            text=text,
            reference_audio_path=str(reference_audio_path),
            language=language
        )
        
        if error:
            return {
                "status": "error",
                "error": error
            }
        
        # Handle output based on storage
        if storage:
            audio_url = _upload_to_s3(audio_bytes, storage)
            if not audio_url:
                return {
                    "status": "error",
                    "error": "Failed to upload audio to S3"
                }
            
            return {
                "status": "success",
                "audio_url": audio_url
            }
        else:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            return {
                "status": "success",
                "audio": audio_b64,
                "format": "wav"
            }


# For local testing
if __name__ == "__main__":
    test_event = {
        "text": "Привет, это тест!",
        "reference_audio_url": "https://example.com/voice.wav",
        "language": "ru"
    }
    result = handler(test_event, None)
    print(f"Result: {result}")
