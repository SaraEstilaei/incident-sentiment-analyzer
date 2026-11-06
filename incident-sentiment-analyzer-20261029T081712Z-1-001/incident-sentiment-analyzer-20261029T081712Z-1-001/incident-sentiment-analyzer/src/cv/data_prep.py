from pathlib import Path
import cv2
import pandas as pd
import numpy as np

def verify_video(video_path: str | Path) -> dict | None:
    path = Path(video_path)
    if not path.exists():
        return None
    
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or fps <= 0 or frame_count < 5:
        return None
        
    return {
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
        "is_valid": True
    }

def extract_uniform_frames(video_path: str | Path, output_dir: Path, video_id: str, num_frames: int = 8, target_size=(224, 224)) -> list[dict]:
    path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    extracted = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        resized = cv2.resize(frame, target_size)
        frame_filename = f"{video_id}_frame_{i:02d}.jpg"
        frame_path = out_dir / frame_filename
        cv2.imwrite(str(frame_path), resized)
        
        extracted.append({
            "video_id": video_id,
            "frame_path": str(frame_path),
            "frame_index": i
        })
        
    cap.release()
    return extracted