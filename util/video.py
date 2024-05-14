from typing import List
from dataclasses import dataclass, field
import os
import numpy as np
import ffmpeg
import log

@dataclass
class VideoState:
    filepath: str = ""
    frames: List[np.ndarray] = field(default_factory=lambda: [])
    width: int = -1
    height: int = -1
    extracted_filepaths: List[str] = field(default_factory=lambda: [])

def read_video(filepath: str) -> VideoState:
    """
    Read a video from the specified filepath
    """
    if not os.path.exists(filepath):
        log.ERROR(f"The specified video path is invalid: {filepath}")
        return None
    

    log.INFO(f"Reading video file: {filepath}")
    state = VideoState()
    state.filepath = filepath

    probe = ffmpeg.probe(filepath)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    state.width = int(video_stream['width'])
    state.height = int(video_stream['height'])

    # extract frames to np.arrays
    log.INFO("Extracting frames")
    out, _ = ffmpeg.input(filepath).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
    state.frames = np.frombuffer(out, np.uint8).reshape([-1, state.height, state.width, 3])
    return state