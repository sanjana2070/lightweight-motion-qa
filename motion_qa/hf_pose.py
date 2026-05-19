# motion_qa/hf_pose.py
# HuggingFace ViTPose pose extraction — replaces MediaPipe in video_pose.py.
# Model: usyd-community/vitpose-base-simple (outputs 17 COCO keypoints).
# Person detection uses YOLOv8n-pose from ultralytics (single pip install).
#
# Output shape: (T, 17, 3)  where dim-2 = (x_norm, y_norm, confidence)
# Joint index → COCO-17 mapping (matches JOINT_GROUPS in modules.py):
#   0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
#   5:l_shoulder  6:l_elbow  7:l_wrist
#   8:r_shoulder  9:r_elbow  10:r_wrist
#   11:l_hip  12:l_knee  13:l_ankle
#   14:r_hip  15:r_knee  16:r_ankle

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from motion_qa.config import POSE_MODEL_ID

_yolo_model = None
_vit_processor = None
_vit_model = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("[hf_pose] Loading YOLOv8n-pose for person detection…")
        _yolo_model = YOLO("yolov8n-pose.pt")  # auto-downloaded on first use
    return _yolo_model


def _load_vitpose():
    global _vit_processor, _vit_model
    if _vit_model is None:
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        print(f"[hf_pose] Loading ViTPose model {POSE_MODEL_ID}…")
        _vit_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID)
        _vit_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID)
        _vit_model.eval()
        if torch.cuda.is_available():
            _vit_model = _vit_model.cuda()
        print("[hf_pose] ViTPose loaded.")
    return _vit_processor, _vit_model


def _extract_keypoints_from_frame(
    frame_rgb: np.ndarray,
    processor,
    vit_model,
    yolo,
) -> Optional[np.ndarray]:
    """
    Detect the primary person and return 17-joint COCO keypoints, or None.
    Returns array of shape (17, 3): (x_norm, y_norm, confidence).
    """
    h, w = frame_rgb.shape[:2]

    # Step 1: detect person bounding boxes with YOLO
    results = yolo(frame_rgb, verbose=False)
    raw_boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # class 0 = person
                raw_boxes.append(box.xyxy[0].cpu().numpy().tolist())

    if not raw_boxes:
        return None

    # Largest box = most prominent person
    largest = max(raw_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = (
        max(0.0, largest[0]), max(0.0, largest[1]),
        min(float(w), largest[2]), min(float(h), largest[3]),
    )
    if x2 <= x1 or y2 <= y1:
        return None

    # Step 2: run ViTPose on the full frame + bounding box.
    # VitPoseImageProcessor requires `boxes` as a required argument —
    # it processes the full image and crops internally.
    from PIL import Image
    pil_img = Image.fromarray(frame_rgb)
    # Format: [batch_of_images → [list_of_person_boxes → [x1,y1,x2,y2]]]
    person_boxes = [[[x1, y1, x2, y2]]]

    inputs = processor(images=pil_img, boxes=person_boxes, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vit_model(**inputs)

    # post_process_pose_estimation returns keypoints in original image pixel coords
    pose_results = processor.post_process_pose_estimation(
        outputs, boxes=person_boxes, threshold=0.0
    )

    # pose_results[image_idx][person_idx]
    if not pose_results or not pose_results[0]:
        return None

    person = pose_results[0][0]
    kps = person["keypoints"].cpu().numpy()     # (17, 2)  pixel coords
    scores = person["scores"].cpu().numpy()      # (17,)

    num_joints = kps.shape[0]
    keypoints = np.zeros((num_joints, 3), dtype=np.float32)
    keypoints[:, 0] = kps[:, 0] / w   # x normalised
    keypoints[:, 1] = kps[:, 1] / h   # y normalised
    keypoints[:, 2] = scores

    return keypoints  # (17, 3)


def extract_pose_from_video(
    video_path: str | Path,
    max_frames: Optional[int] = 300,
    target_fps: float = 10.0,
    max_width: int = 640,
    model_id: str = POSE_MODEL_ID,
) -> tuple[np.ndarray, float]:
    """
    Extract 17-joint COCO pose trajectory from a video using ViTPose.

    Parameters
    ----------
    max_frames : int
        Hard cap on frames processed (after stride is applied).
    target_fps : float
        Downsample to this many frames per second before inference.
        10 fps is enough for freeze/rhythm/displacement analysis.
    max_width : int
        Resize frames to at most this width before inference (keeps aspect ratio).
        Reduces YOLO + ViTPose CPU time significantly at high resolutions.

    Returns
    -------
    motion : np.ndarray, shape (T, 17, 3), dtype float32
        Third channel: (x_norm, y_norm, confidence).
    fps : float
        Effective frames per second after downsampling (= target_fps).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 240:
        src_fps = 30.0

    # Stride: process every Nth frame to approximate target_fps
    stride = max(1, int(round(src_fps / target_fps)))
    effective_fps = src_fps / stride
    print(f"[hf_pose] src_fps={src_fps:.1f}, stride={stride}, "
          f"effective_fps={effective_fps:.1f}, max_width={max_width}")

    processor, vit_model = _load_vitpose()
    yolo = _load_yolo()

    frames: list[np.ndarray] = []
    last_valid: Optional[np.ndarray] = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # OpenCV can return ret=True with frame=None at a truncated file boundary
        if not ret or frame is None:
            break

        frame_idx += 1

        # Skip frames to hit target_fps
        if (frame_idx - 1) % stride != 0:
            continue

        if max_frames is not None and len(frames) >= max_frames:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            continue

        # Resize to reduce inference time on high-res video (e.g. 1920x1080 → 640x360)
        h, w = frame_rgb.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        kps = _extract_keypoints_from_frame(frame_rgb, processor, vit_model, yolo)

        if kps is not None:
            last_valid = kps
            frames.append(kps)
        elif last_valid is not None:
            frames.append(last_valid.copy())

    cap.release()

    if not frames:
        raise ValueError(
            f"Could not extract any pose data from {video_path}. "
            "The file may be corrupted or too short. "
            "Try re-uploading, or check that a person is clearly visible in the clip."
        )

    motion = np.stack(frames, axis=0).astype(np.float32)  # (T, 17, 3)
    print(f"[hf_pose] Extracted {len(frames)} frames → motion shape {motion.shape}")
    return motion, effective_fps
