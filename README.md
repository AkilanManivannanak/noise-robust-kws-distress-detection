# Noise-Robust Keyword Spotting for Distress Detection (KWS + Noise Robustness)

A practical **distress keyword spotting (KWS)** pipeline that detects emergency/distress phrases from speech under **real-world noisy conditions**. The system is designed for environments where background noise is unavoidable (public spaces, streets, vehicles, crowded rooms) and the cost of missed detections is high.

This repo includes:
- A full end-to-end implementation in **`ASR.ipynb`** (data prep → inference/training → evaluation).  
- A final report: **`fpr-Team-6-2.pdf`**.  

> Note: Raw datasets are **not** included in the repository (links only). :contentReference[oaicite:1]{index=1}

---

## Why this matters (project value)

Most voice systems fail in noise. This project targets two failure modes:
1. **False negatives**: distress speech is missed due to noise masking.
2. **False positives**: background noise triggers distress keywords.

The goal is **robust recall without exploding false alarms**, using a reproducible pipeline + evaluation metrics that recruiters can trust.

---

## Problem statement

Given an audio clip, detect whether it contains **distress keywords/phrases** (examples: “help”, “call police”, “emergency”, etc.) even when:
- the speaker is far from the mic,
- there is background noise,
- pronunciation varies by speaker.

**Output**
- `distress_detected: true/false`
- `matched_keywords: [...]`
- (optional) transcript + confidence scores

---

## Data sources (no raw data in repo)

### 1) Team-recorded voices (primary “real” samples)
We recorded **3 voice samples** ourselves (team voices). These represent **in-the-wild variability** (speaker timbre, accent, mic distance).

- Stored privately in Google Drive:  
  `DATA_PRIVATE_DRIVE_FOLDER = https://drive.google.com/drive/u/2/folders/1Da_k5LG4LMmz4CL58PRI6mqnK8crfQVq`

### 2) Kaggle-sourced speech data (majority of samples)
All other audio samples used for keyword/background/noise augmentation were sourced from **Kaggle datasets**.

> Replace the placeholders below with the exact Kaggle dataset pages you used:
- Kaggle dataset #1: `<KAGGLE_DATASET_LINK_1>`
- Kaggle dataset #2: `<KAGGLE_DATASET_LINK_2>`
- (optional) noise/background dataset: `<KAGGLE_NOISE_DATASET_LINK>`

### Licensing / sharing
- This repo **does not redistribute** Kaggle audio files.  
- Users must download datasets from the original sources and place them locally.

---

## Architecture

This project supports two common robust KWS patterns. Your notebook/report implements **one** of these—mark the correct one:

### Option A — ASR → Keyword Matching (most practical)
1. **Preprocess audio** (resample, normalize, trim/split)
2. Run a **pretrained ASR model** to get transcript
3. Apply **keyword/phrase matching** + confidence/threshold rules
4. Report **distress detection** + matched keywords

**ASR backbone (fill in from notebook):** `<Whisper / Wav2Vec2 / SpeechRecognition / other>`

### Option B — Direct KWS Classifier (small footprint)
1. Convert audio → **log-mel spectrogram / MFCC**
2. Train/infer a **CNN/CRNN** classifier to detect keywords directly
3. Apply post-processing (smoothing, thresholds)

**Classifier backbone (fill in):** `<CNN / CRNN / DS-CNN / other>`

---

## Workflow (end-to-end)

```text
Audio (.wav/.mp3)
  ↓
Preprocessing
  - resample (e.g., 16kHz)
  - normalize loudness
  - trim silence / chunking
  ↓
Noise Robustness
  - noise mixing / augmentation (optional)
  - SNR evaluation (optional)
  ↓
Model Inference
  - ASR transcript OR KWS classifier
  ↓
Decision Layer
  - keyword matching + thresholds
  - final distress_detected flag
  ↓
Evaluation
  - Precision / Recall / F1
  - Confusion matrix
  - (optional) WER + keyword recall

```
# Metrics 
For distress detection, accuracy alone is weak. You should report:

  
1. Core classification metrics
      - Precision (controls false alarms)
      - Recall (controls missed distress)
      - F1-score (balance)
      - Confusion matrix

2. Noise-robustness metrics 

Evaluate performance under different noise levels:

     - F1 @ clean
     - F1 @ noisy (or at SNR buckets like 20/10/5/0 dB)
     - False alarm rate (FPR) at a fixed recall target (optional)

---
# Results
