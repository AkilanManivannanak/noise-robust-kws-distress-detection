![Cover](noise_cover_image.png)


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
## Results

### Dataset split (fixed across KWS experiments)
- **Train:** 84,843 examples  
- **Validation:** 9,981 examples  
- **Test:** 11,005 examples  

---

### KWS (11-class) — Clean-test performance

| Variant | Features | Robustness Training | Clean Test Loss | Clean Test Accuracy |
|---|---|---|---:|---:|
| **CLEAN-LOGMEL** | log-Mel | No | **0.4002** | **87.16%** |
| **NOISE-LOGMEL** | log-Mel | Yes (noise mixing) | **0.4780** | **84.18%** |
| **SPECAUG-LOGMEL** | log-Mel | Yes (noise) + SpecAug | **0.6330** | **78.57%** |
| **MFCC-NOISE** | MFCC | Yes (noise mixing) | **0.3901** | **88.37%** |

**Best clean accuracy:** **MFCC-NOISE (88.37%)**

---

### KWS — Noise robustness (Accuracy vs SNR)

Accuracy (%) measured under clean audio and noisy audio at different SNR levels.

| Variant | Clean | 20 dB | 10 dB | 0 dB |
|---|---:|---:|---:|---:|
| **CLEAN-LOGMEL** | **87.16%** | 83.69% | 75.91% | 67.75% |
| **NOISE-LOGMEL** | 84.18% | 82.62% | **79.37%** | **71.60%** |
| **SPECAUG-LOGMEL** | 78.57% | 78.56% | 75.72% | 67.63% |
| **MFCC-NOISE** | **88.37%** | **88.04%** | **85.29%** | **77.02%** |

**Key takeaways**
- **Noise mixing improves low-SNR robustness:** at **0 dB**, NOISE-LOGMEL (71.60%) > CLEAN-LOGMEL (67.75%).
- **MFCC-NOISE is strongest overall:** best clean accuracy and best performance under heavy noise (**77.02% at 0 dB**).
- **SpecAug (as configured here) underperformed** both clean and noisy conditions compared to the other variants.

---

### KWS — Detailed metrics (SpecAug log-Mel on clean test)

> Important: The test set is **highly imbalanced** (`unknown` has 6,931/11,005 samples), which inflates overall accuracy. Macro averages better reflect performance across the command classes.

**Overall**
- **Accuracy:** 0.79  
- **Macro avg (P/R/F1):** 0.70 / 0.62 / 0.62  
- **Weighted avg (P/R/F1):** 0.80 / 0.79 / 0.78  

<details>
<summary><b>Per-class classification report (SpecAug log-Mel)</b></summary>

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| yes | 0.94 | 0.83 | 0.88 | 419 |
| no | 0.75 | 0.65 | 0.69 | 405 |
| up | 0.48 | 0.84 | 0.61 | 425 |
| down | 0.84 | 0.30 | 0.44 | 406 |
| left | 0.62 | 0.83 | 0.71 | 412 |
| right | 0.85 | 0.34 | 0.49 | 396 |
| on | 0.58 | 0.54 | 0.56 | 396 |
| off | 0.51 | 0.82 | 0.63 | 402 |
| stop | 0.70 | 0.39 | 0.50 | 411 |
| go | 0.61 | 0.34 | 0.43 | 402 |
| unknown | 0.87 | 0.90 | 0.88 | 6931 |

</details>

<details>
<summary><b>Confusion matrix (SpecAug log-Mel, clean test) — counts</b></summary>

Labels: `yes, no, up, down, left, right, on, off, stop, go, unknown`

```text
[[ 348    0    0    0   16    0    0   15    0    0   40]
 [   6  262    3    2    4    0    0    4    0    6  118]
 [   0    0  359    0    1    0    0   40    3    0   22]
 [   0   16    2  123    0    0   17    0    0   60  188]
 [   2    0    6    0  340    1    0   10    1    0   52]
 [   1    0    1    0  116  136    0    0    1    0  141]
 [   0    0    8    0    0    0  214   41    0    0  133]
 [   0    0   50    0    0    0    4  330    0    0   18]
 [   0    0  148    0    0    0    0    6  159    0   98]
 [   0   58    8   14    1    0   12   14    0  135  160]
 [  13   14  166    8   71   23  124  188   63   20 6241]
```
Notable weak classes (SpecAug log-Mel)

- Low recall: down (0.30), right (0.34), go (0.34), stop (0.39).
- Many commands are misclassified as unknown (dominant class), consistent with the imbalance.

Distress (binary) — Single-task classifier results

Test set size: 180 samples
   - non_distress: 132
   - distress: 48

Final test
   
   - Loss: 0.5824
   - Accuracy: 73.33%

Confusion matrix (counts)

Rows = true, columns = predicted ([non_distress, distress]):

[[131   1]
 [ 47   1]]


Classification Report 

| Class            | Precision | Recall |   F1 | Support |
| ---------------- | --------: | -----: | ---: | ------: |
| non_distress     |      0.74 |   0.99 | 0.85 |     132 |
| distress         |      0.50 |   0.02 | 0.04 |      48 |
| **macro avg**    |      0.62 |   0.51 | 0.44 |     180 |
| **weighted avg** |      0.67 |   0.73 | 0.63 |     180 |


# Interpretation

The model is not usable for distress detection in its current form: distress recall = 0.02 (it detects 1 out of 48 distress samples).
The 73% accuracy is misleading due to class imbalance and the model defaulting to non_distress.

# Multi-task (KWS + Distress) — Final test accuracy

    - Final KWS test accuracy: 62.98%
    - Final Distress test accuracy: 73.33%


Model efficiency (latency + size)

Measured on Apple MPS:

- Forward (model-only) average: 0.443 ms
- Parameters: 111,051
- State dict size: 0.43 MB
- End-to-end (feature extraction + forward) avg: 1.17 ms
- End-to-end p95: 2.18 ms
---
