# Reflection Mirror Prototype

Streamlit prototype for **expression-informed reflection** using [Hume](https://www.hume.ai/) Expression Measurement. You can analyze pasted text or uploaded media, review top emotion-style signals, read a lightweight “possible read” summary, compare two session runs, and export raw JSON.

> This tool surfaces **possible observer-facing interpretations of expression**, not verified internal emotion. Use results thoughtfully and in appropriate contexts.

## Features

- **Text reflection** — Send text through language models (`sentence` granularity) and inspect segment-level signals.
- **Audio / video / image reflection** — Upload supported files and optionally enable **language**, **prosody**, **vocal burst**, and **facial expression** analysis (combinations that do not apply to still images show a warning in the app).
- **Compare** — Side-by-side view of two saved runs plus a simple signal comparison table.
- **Saved runs** — In-session history with expanders and per-run JSON download.

## Requirements

- Python 3.10+ recommended  
- A **Hume API key** with access to Expression Measurement

## Setup

### 1. Clone and install

```bash
git clone https://github.com/wtrmrr/reflection_mirror_prototype.git
cd reflection_mirror_prototype
pip install -r requirements.txt
```

### 2. API key

Set `HUME_API_KEY` in either of these ways:

**Environment variable (terminal session):**

```bash
export HUME_API_KEY="your_key_here"
```

**Streamlit secrets (local dev):** create `.streamlit/secrets.toml` (do not commit this file):

```toml
HUME_API_KEY = "your_key_here"
```

This repo’s `.gitignore` ignores `.streamlit/secrets.toml` by default.

### 3. Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Supported uploads

| Type | Extensions |
|------|------------|
| Audio | `.mp3`, `.wav`, `.m4a` |
| Video | `.mp4`, `.mov` |
| Image | `.jpg`, `.jpeg`, `.png` |

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI and Hume streaming integration |
| `requirements.txt` | Python dependencies |

## License

No license is specified in this repository; add one if you intend others to reuse the code.
