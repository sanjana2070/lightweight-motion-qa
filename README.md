# Dance QA

A project that lets you **ask questions about dance videos** through a web interface. Upload a short clip — the system classifies the dance genre using **X-CLIP** (zero-shot video-text matching), extracts a 17-joint COCO skeleton using **ViTPose + YOLOv8**, runs dance-specific analysis modules in PyTorch, and uses either a **local HuggingFace LLM** or a fast heuristic to phrase the final answer.

Trained on **AIST++** (10 dance genres: Breaking, Popping, Locking, Hip-Hop, House, Waacking, Krump, Street Jazz, Ballet Jazz).

---

## What can I ask?

Upload a short dance video and ask questions like:

| Category | Example questions |
|---|---|
| **Genre / Style** | "What dance style is this?" · "Is this Hip-Hop or House?" · "Is this waacking or locking?" |
| **Freeze** | "Does the performer freeze at any point?" · "How many freezes are in this clip?" · "How long is the freeze?" |
| **House Groove** | "Is jacking present in this clip?" · "Does the performer show House groove?" |
| **Rhythm** | "How rhythmically regular is the movement?" · "Is the dancing consistent in tempo?" |
| **Floor Coverage** | "Is the performer dancing in place or traveling?" · "How much floor space do they cover?" · "Which direction does the performer move?" |
| **Body / Limb** | "Which body part is most active?" · "Is this upper body or lower body dominated?" |
| **Duration** | "How long is this clip in seconds?" |

> **Honest limitation:** genre-level classification only (Breaking, House, Waacking, etc.). Step-level recognition ("is this a toprock?") requires a trained step classifier — planned for a future phase.

---

## Architecture

### `motion_qa/`

| File | Role |
|---|---|
| `hf_video.py` | X-CLIP zero-shot dance genre classification from raw video frames |
| `hf_pose.py` | ViTPose + YOLOv8n-pose → `(T, 17, 3)` COCO-17 skeleton per frame |
| `hf_llm.py` | Local LLM inference (Phi-3-mini-4k-instruct, 4-bit on GPU / float32 on CPU) |
| `modules.py` | Analysis tools: `classify_dance_style`, `detect_freeze`, `detect_jacking`, `compute_rhythm_regularity`, `dominant_direction`, `global_displacement`, `displacement_category`, `most_active_limb`, `clip_duration` |
| `registry.py` | Central `MODULE_MAP` connecting tool names to module functions |
| `planner.py` | Routes questions to the right tool (heuristic keyword matching or Phi-3 LLM) |
| `answerer.py` | Formats tool output into readable text (rule-based or Phi-3 LLM) |
| `features.py` | `compute_basic_features` — displacement, per-joint path length, speed |
| `video_pose.py` | Thin shim: delegates to `hf_pose.py` |
| `datasets.py` | Dataset loader compatible with AIST++ and any preprocessed `metadata.json` |
| `config.py` | Env var config (`USE_LLM`, `HF_MODEL_ID`, `POSE_MODEL_ID`, `XCLIP_MODEL_ID`) |

### `scripts/`

| File | Role |
|---|---|
| `app_web.py` | Gradio web app — upload video, ask a question, get an answer |
| `cli_app.py` | Interactive CLI for browsing the AIST++ dataset and running Q&A |
| `preprocess_aist.py` | Converts AIST++ videos → `(T, 17, 3)` pose arrays + `metadata.json` |
| `preprocess_dance_dataset.py` | Same pipeline for custom Hip-Hop / House video collections |
| `train_style_classifier.py` | Fine-tunes X-CLIP on your preprocessed AIST++ clips for better accuracy |
| `run_demo.py` | Quick one-clip terminal demo |

---

## Pipeline

```
User uploads video
       ↓
Planner routes question to tool
       ↓
  ┌──────────────────────────────────────────────┐
  │ classify_dance_style → hf_video.py (X-CLIP)  │  raw video frames
  │ detect_freeze        → hf_pose.py + modules  │  joint velocity
  │ detect_jacking       → hf_pose.py + modules  │  FFT on hip y
  │ compute_rhythm_regularity → modules           │  autocorrelation
  │ displacement / direction  → modules           │  root trajectory
  │ most_active_limb          → modules           │  per-joint path length
  │ clip_duration             → modules           │  T / fps
  └──────────────────────────────────────────────┘
       ↓
Answerer (rule-based or Phi-3-mini LLM)
       ↓
Answer text → Gradio UI
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages: `transformers`, `torch`, `ultralytics`, `decord`, `gradio`, `opencv-python`, `Pillow`, `accelerate`, `bitsandbytes`.

> `bitsandbytes` enables 4-bit quantization of Phi-3 on CUDA GPUs. CPU fallback runs in float32 (slower). `mediapipe` is not required.

### 3. Configure via `.env`

```bash
# true  → use local Phi-3 LLM for planning + answering (downloads ~4 GB on first run)
# false → fast offline heuristic mode, no model download needed
USE_LLM=false

# Optional overrides (defaults shown):
# HF_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
# POSE_MODEL_ID=usyd-community/vitpose-base-simple
# XCLIP_MODEL_ID=microsoft/xclip-base-patch32
```

X-CLIP (~600 MB) is downloaded on the first style classification query. No API key required.

---

## Data setup

### AIST++ (primary dataset)

1. Download AIST++ videos from [google.github.io/aistplusplus_dataset](https://google.github.io/aistplusplus_dataset/)
2. Place them in `data/aist_raw/videos/` (filenames must follow the AIST++ convention, e.g. `mBR_sFM_cAll_d04_mBR0_ch01.mp4`)
3. Run:

```bash
python -m scripts.preprocess_aist
```

This extracts ViTPose skeletons and generates Q&A pairs per clip:

```
data/aist/motions/*.npy      # (T, 17, 3) pose arrays
data/aist/metadata.json      # dataset manifest
```

### Custom dance clips (Hip-Hop / House)

Place videos in:
```
data/raw_videos/hip_hop/
data/raw_videos/house/
```

Optionally add a `.json` annotation alongside each video:
```json
{"move_labels": [{"label": "toprock", "start_frame": 0, "end_frame": 90}]}
```

Then run:
```bash
python -m scripts.preprocess_dance_dataset
```

---

## Fine-tuning X-CLIP (optional)

Zero-shot X-CLIP works out of the box. For better accuracy after preprocessing AIST++:

```bash
python -m scripts.train_style_classifier
```

Saves weights to `models/xclip_dance_finetuned/`. To use them, add to `.env`:

```bash
XCLIP_MODEL_ID=models/xclip_dance_finetuned
```

---

## Running

```bash
# Web app
python -m scripts.app_web

# Interactive CLI (requires preprocessed AIST++ data)
python -m scripts.cli_app

# Quick one-clip terminal demo
python -m scripts.run_demo
```

---

## Future directions

Step-level recognition (toprock, 6-step, jacking footwork, etc.) is the next planned phase — requires a trained classifier on labeled per-step clips with front and side camera views.
