from .vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)

# Add audio utilities
from .audio_process import fetch_audio

# Export all functions
__all__ = [
    "extract_vision_info",
    "fetch_image", 
    "fetch_video",
    "process_vision_info",
    "smart_resize",
    "fetch_audio",
]