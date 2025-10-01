from typing import Dict, Any, Tuple
import numpy as np

def fetch_audio(ele: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    Fetch and process audio data from HuggingFace dataset format.
    
    Args:
        audio: Dictionary with 'bytes' key containing raw audio data
               Example: {"bytes": b"...", "path": "..."}
    
    Returns:
        Tuple of (audio_array, sample_rate)
            - audio_array: numpy array of audio samples (float32)
            - sample_rate: sampling rate in Hz
    """
    import soundfile as sf
    import io

    # get audio bytes
    audio_bytes = ele["bytes"]

    # convert bytes to numpy array
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return audio_array, sample_rate