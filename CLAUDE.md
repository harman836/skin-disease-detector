# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Flask web app for skin disease detection. It classifies skin lesions into 5 conditions (Melanoma, Basal Cell Carcinoma, Eczema, Psoriasis, Tinea Ringworm) using a YOLOv8 model exported to ONNX, with a Gemini-powered chatbot for follow-up questions.

## Running the App

```bash
# Activate venv first
source venv/Scripts/activate  # Windows Git Bash
# or
venv\Scripts\activate.bat     # Windows CMD

pip install -r requirements.txt
python app.py
```

The app runs at `http://localhost:5000` in debug mode.

## Environment Setup

Requires a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

The model file `yolov8model.onnx` must be present in the project root. It is loaded at startup and is required for inference.

## Architecture

**`app.py`** — single-file Flask backend with all routes and logic:

- `/upload` (POST) — receives image, runs `prepare_image()` (EXIF fix + CLAHE contrast enhancement + letterbox to 224×224), saves to `uploads/`, spawns a background thread for inference, returns a data URL preview
- `/progress/<filename>` — polling endpoint; frontend polls this until `completed: true`; background thread writes status into the global `TASKS` dict
- `/heatmap/<filename>` — generates an occlusion sensitivity heatmap via a 7×7 grid occlusion sweep (49 ONNX forward passes); slow, called lazily from the results page
- `/chat` (POST) — forwards user message + image + analysis scores to Gemini 2.5 Flash with a dermatology-scoped system prompt

**Inference flow:**
1. `run_onnx_inference()` — file path → PIL → numpy `[1,3,224,224]` float32 → ONNX → softmax → class scores
2. `run_onnx_inference_array()` — same but accepts a pre-loaded numpy array (used by heatmap generation)
3. Class names are read from the ONNX model's custom metadata (`names` key) at startup

**Frontend** (`templates/` + `static/`):
- Multi-page flow: `index.html` → upload → `loading.html` (polls `/progress`) → `results.html`
- `capture.html` supports live camera capture via `getUserMedia`
- `static/common.js` — shared JS; `static/styles.css` — shared CSS
- Results page fetches the heatmap after load and renders a chatbot panel

**`preprocessing.py`** — standalone script / notebook cells for offline experimentation with the `.pt` model (not used by the Flask app at runtime; the app uses ONNX directly).

**`SkinDiseaseScreening_CNN_Models.ipynb`** — Jupyter notebook for model training and evaluation.

## Design Context

### Users
Everyday people (patients, general public) checking a suspicious skin spot — likely on a phone, possibly anxious, with no medical training. Mobile is the primary context. Anxiety is the default emotional state on arrival.

### Brand Personality
**Three words: Trustworthy. Empowering. Precise.**

The voice is calm, direct, and knowledgeable — like a trusted GP summarizing a test result, not a chatbot performing friendliness.

### Aesthetic Direction
Clinical & trustworthy. Clean surfaces, restrained typography, purposeful whitespace. Existing palette (navy `#2a4b7c`, teal `#489fb5`, warm cream `#faf8f5`) and type stack (Fraunces + Inter) are correct. Dark mode is desirable but not blocking.

**Must NOT look like:** a generic AI chatbot (dark sidebar + floating bubbles + robot avatar) or a consumer wellness app (pastel gradients, gamified elements).

### Design Principles
1. **Clarity over cleverness.** Every interaction must be immediately understandable by a stressed, non-technical person.
2. **Information hierarchy earns trust.** Primary diagnosis dominates. Supporting data follows. Users should never hunt for the answer.
3. **Actions over information.** Every result must conclude with a clear, actionable recommendation.
4. **Restraint signals professionalism.** Fewer animations, no decorative gradients, no personality-driven micro-copy.
5. **Mobile-first, anxiety-aware.** Generous touch targets, legible in bright daylight, informative loading states.

### Accessibility Target
Best-effort WCAG AA — fix obvious contrast, keyboard nav, and screen reader issues. Full WCAG compliance is not a blocker for this research project.

> Full design context in `.impeccable.md` at the project root.
