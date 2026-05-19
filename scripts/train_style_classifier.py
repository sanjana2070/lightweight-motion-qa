# scripts/train_style_classifier.py
#
# Fine-tune X-CLIP on the preprocessed AIST++ dataset for dance genre classification.
# The zero-shot X-CLIP already works without running this script.
# Run this to improve accuracy once you have labeled AIST++ clips in data/aist/.
#
# HOW TO USE:
#   1. Run preprocess_aist.py first to build data/aist/metadata.json
#   2. Run:  python -m scripts.train_style_classifier
#
# Saves fine-tuned weights to:  models/xclip_dance_finetuned/
#
# Requirements: GPU with >=8 GB VRAM recommended for fine-tuning.
# On CPU the training loop will run but will be very slow.

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2

from motion_qa.config import XCLIP_MODEL_ID, AIST_GENRE_LABELS

META_PATH = Path("data/aist/metadata.json")
OUTPUT_DIR = Path("models/xclip_dance_finetuned")
NUM_FRAMES = 8
BATCH_SIZE = 4
NUM_EPOCHS = 5
LR = 1e-5


class VideoStyleDataset(Dataset):
    """Loads video frames + genre label for X-CLIP fine-tuning."""

    def __init__(self, items: list[dict], processor) -> None:
        # Only keep items that have a video_path and a classify_dance_style answer
        self.samples: list[dict] = []
        for item in items:
            video_path = item.get("video_path", "")
            if not video_path or not Path(video_path).exists():
                continue
            for q in item.get("questions", []):
                if q.get("type") == "classify_dance_style":
                    genre = q.get("a", "")
                    if genre in AIST_GENRE_LABELS:
                        self.samples.append({
                            "video_path": video_path,
                            "label_idx": AIST_GENRE_LABELS.index(genre),
                        })
                    break

        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = _sample_frames(sample["video_path"], NUM_FRAMES)
        inputs = self.processor(
            videos=frames,
            return_tensors="pt",
            padding=True,
        )
        # Remove batch dim added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["label"] = torch.tensor(sample["label_idx"], dtype=torch.long)
        return inputs


def _sample_frames(video_path: str, num_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = set(int(i * total / num_frames) for i in range(num_frames))

    frames: list[Image.Image] = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
    return frames[:num_frames]


def collate_fn(batch: list[dict]) -> dict:
    keys = [k for k in batch[0] if k != "label"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["label"] = torch.stack([b["label"] for b in batch])
    return out


def main() -> None:
    if not META_PATH.exists():
        print(f"[error] {META_PATH} not found. Run preprocess_aist.py first.")
        return

    with open(META_PATH, encoding="utf-8") as f:
        items = json.load(f)

    from transformers import XCLIPProcessor, XCLIPModel

    print(f"[train] Loading X-CLIP model {XCLIP_MODEL_ID}…")
    processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID)
    model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    model = model.to(device)

    # Build dataset — shuffle and split 80/20
    random.shuffle(items)
    split = int(0.8 * len(items))
    train_ds = VideoStyleDataset(items[:split], processor)
    val_ds = VideoStyleDataset(items[split:], processor)
    print(f"[train] {len(train_ds)} train, {len(val_ds)} val samples.")

    if len(train_ds) == 0:
        print("[error] No usable training samples. Check video_path fields in metadata.json.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Text embeddings for all genre labels (frozen during training)
    text_inputs = processor(
        text=[f"a person performing {g.lower()} dance" for g in AIST_GENRE_LABELS],
        return_tensors="pt",
        padding=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("label").to(device)
            video_inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**video_inputs, **text_inputs)
            logits = outputs.logits_per_video  # (B, num_labels)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("label").to(device)
                video_inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**video_inputs, **text_inputs)
                preds = outputs.logits_per_video.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"  Epoch {epoch}/{NUM_EPOCHS} — "
              f"loss: {total_loss/len(train_loader):.4f}, "
              f"val_acc: {val_acc*100:.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\n[done] Fine-tuned model saved to {OUTPUT_DIR}/")
    print("Set XCLIP_MODEL_ID=models/xclip_dance_finetuned in .env to use it.")


if __name__ == "__main__":
    main()
