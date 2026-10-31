from pathlib import Path
import cv2
import numpy as np

def verify_video(video_path: str | Path) -> dict | None:
    video_path = Path(video_path)
    if not video_path.exists() or video_path.stat().st_size == 0:
        return None
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Try reading the first frame successfully
    ret, first_frame = cap.read()
    if not ret or first_frame is None or fps <= 0 or frame_count < 5:
        cap.release()
        return None
        
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    
    return {
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": int(frame_count),
        "is_valid": True
    }

def extract_uniform_frames(
    video_path: str | Path,
    output_dir: Path,
    video_id: str,
    num_frames: int = 8,
    target_size: tuple[int, int] = (224, 224)
) -> list[dict]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        cap.release()
        return []

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    saved_frames = []

    vid_output_dir = output_dir / video_id
    vid_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        # Resize to target size (224, 224)
        resized_frame = cv2.resize(frame, target_size)
        frame_filename = vid_output_dir / f"frame_{idx:03d}.jpg"
        cv2.imwrite(str(frame_filename), resized_frame)
        
        saved_frames.append({
            "video_id": video_id,
            "frame_path": str(frame_filename),
            "frame_idx": int(frame_idx)
        })

    cap.release()
    return saved_frames