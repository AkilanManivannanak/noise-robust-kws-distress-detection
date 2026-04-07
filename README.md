<div align="center">

![noise_cover_image](./noise_cover_image.png)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7a2e00,50:ff7b00,100:ffb347&height=220&section=header&text=%F0%9F%8E%99%EF%B8%8F%20Noise-Robust%20Keyword%20Spotting%20for%20Distress%20Detection&fontSize=50&fontColor=ffffff&fontAlignY=38&desc=Multimodal%20AI%20Driver%20Safety%20System&descAlignY=58&descSize=22&animation=fadeIn" width="100%"/>
# 🎙️ Noise-Robust Keyword Spotting for Distress Detection

**Live repo:** [github.com/AKilalours/noise-robust-kws-distress-detection](https://github.com/AKilalours/noise-robust-kws-distress-detection)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![p95 Latency](https://img.shields.io/badge/p95%20latency-2.18ms-brightgreen?style=flat-square)]()
[![Model Size](https://img.shields.io/badge/model%20size-0.43MB-brightgreen?style=flat-square)]()
[![Best Accuracy](https://img.shields.io/badge/best%20accuracy-88.37%25-blue?style=flat-square)]()

</div>

---

## 🎯 Goal & SLOs

Detect **distress keywords** ("help", "call police", "emergency") in real-world noisy audio — where false negatives cost safety.


| SLO | Target | Achieved |
|---|---|---|
| **p95 end-to-end latency** | < 5 ms | **2.18 ms** ✅ |
| **Model size** | < 1 MB (edge-deployable) | **0.43 MB** ✅ |
| **Clean accuracy (KWS)** | > 85% | **88.37% (MFCC-NOISE)** ✅ |
| **0 dB SNR accuracy** | > 70% | **77.02%** ✅ |
| **Distress recall** | > 80% | ❌ 0.02 (active improvement — see postmortem) |

---

## 📐 Architecture

```
Audio Input (.wav / .mp3)
        │
        ▼
┌───────────────────────────────────────────────┐
│             PREPROCESSING                     │
│  • Resample → 16 kHz                          │
│  • Loudness normalization (peak / RMS)        │
│  • Silence trim + 1-sec chunking              │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│         NOISE AUGMENTATION (training)         │
│  • Mix signal + background noise at random SNR│
│  • SNR buckets: 0 / 5 / 10 / 20 dB            │
│  • SpecAugment (freq + time masking) variant  │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│           FEATURE EXTRACTION                  │
│  • Log-Mel spectrogram  (40 bins, 25ms frame) │
│  • MFCC (40 coeffs, delta + delta-delta)      │
└───────────────────┬───────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌────────────────┐   ┌────────────────────────┐
│  KWS HEAD      │   │  DISTRESS HEAD         │
│  CNN/CRNN      │   │  Binary classifier     │
│  11-class      │   │  distress / non        │
└────────┬───────┘   └──────────┬─────────────┘
         └──────────┬───────────┘
                    │ Multi-task loss
                    ▼
┌───────────────────────────────────────────────┐
│           DECISION LAYER                      │
│  • Keyword match + confidence threshold       │
│  • distress_detected: true / false            │
│  • matched_keywords: [...]                    │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│           EVALUATION                          │
│  • Precision / Recall / F1 per class          │
│  • Confusion matrix                           │
│  • Accuracy @ SNR: clean / 20 / 10 / 0 dB     │
└───────────────────────────────────────────────┘
```

**Trade-offs considered:**
- **MFCC vs Log-Mel:** MFCC is more compact and more robust to additive noise → chosen for production path.
- **SpecAugment:** improves generalization in large-data regimes, but hurt this small dataset — dropped in best model.
- **Multi-task (KWS + Distress) vs single-task:** multi-task converged at lower KWS accuracy (62.98%) vs single-task (88.37%). Kept single-task KWS as primary; multi-task is experimental.
- **Latency vs quality:** No reranker or ensemble — single lightweight CNN keeps p95 at 2.18 ms. Accuracy is the cost.

---

## Why This Matters

Most voice systems fail in noise. This project targets two failure modes:

1. **False negatives:** distress speech is missed due to noise masking.
2. **False positives:** background noise triggers distress keywords.

Goal: **robust recall without exploding false alarms**, using a reproducible pipeline and evaluation metrics calibrated for safety-critical use cases.

---

## Data Sources

> Raw datasets are **not** included (links only).

### 1) Team-recorded voices
3 voice samples recorded by team members — representing in-the-wild variability (timbre, accent, mic distance).

- Private Drive: `https://drive.google.com/drive/u/2/folders/1Da_k5LG4LMmz4CL58PRI6mqnK8crfQVq`

### 2) Kaggle-sourced speech data
All other audio (keywords, backgrounds, noise augmentation) sourced from Kaggle.

- Kaggle dataset #1: `<KAGGLE_DATASET_LINK_1>`
- Kaggle dataset #2: `<KAGGLE_DATASET_LINK_2>`
- Noise/background dataset: `<KAGGLE_NOISE_DATASET_LINK>`

### Dataset split (fixed across experiments)
| Split | Samples |
|---|---|
| Train | 84,843 |
| Validation | 9,981 |
| Test | 11,005 |

---

## Results

### KWS (11-class) — Clean-test performance

| Variant | Features | Robustness Training | Clean Loss | Clean Accuracy |
|---|---|---|---:|---:|
| CLEAN-LOGMEL | log-Mel | No | 0.4002 | 87.16% |
| NOISE-LOGMEL | log-Mel | Noise mixing | 0.4780 | 84.18% |
| SPECAUG-LOGMEL | log-Mel | Noise + SpecAug | 0.6330 | 78.57% |
| **MFCC-NOISE** | **MFCC** | **Noise mixing** | **0.3901** | **88.37% ✅** |

### KWS — Noise robustness (Accuracy vs SNR)

| Variant | Clean | 20 dB | 10 dB | 0 dB |
|---|---:|---:|---:|---:|
| CLEAN-LOGMEL | 87.16% | 83.69% | 75.91% | 67.75% |
| NOISE-LOGMEL | 84.18% | 82.62% | 79.37% | 71.60% |
| SPECAUG-LOGMEL | 78.57% | 78.56% | 75.72% | 67.63% |
| **MFCC-NOISE** | **88.37%** | **88.04%** | **85.29%** | **77.02%** |

**Key takeaways:**
- Noise mixing improves low-SNR robustness: at 0 dB, NOISE-LOGMEL (71.60%) > CLEAN-LOGMEL (67.75%).
- MFCC-NOISE is strongest overall — best clean and best heavy-noise performance.
- SpecAugment as configured underperformed both conditions (insufficient data volume to benefit).

### KWS — Detailed metrics (SpecAug log-Mel, clean test)

> Test set is **highly imbalanced** (`unknown` = 6,931/11,005). Macro averages better reflect real class performance.

| Metric | Value |
|---|---|
| Accuracy | 0.79 |
| Macro avg P/R/F1 | 0.70 / 0.62 / 0.62 |
| Weighted avg P/R/F1 | 0.80 / 0.79 / 0.78 |

<details>
<summary><b>Per-class classification report</b></summary>

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
<summary><b>Confusion matrix (SpecAug log-Mel, clean test)</b></summary>

Labels: `yes, no, up, down, left, right, on, off, stop, go, unknown`

```
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
 [  13   14  166    8   71   23  124  188   63   20 6241]]
```

</details>

### Distress (binary) — Single-task classifier

| Metric | Value |
|---|---|
| Test set size | 180 (132 non-distress / 48 distress) |
| Loss | 0.5824 |
| Accuracy | 73.33% |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| non_distress | 0.74 | 0.99 | 0.85 | 132 |
| distress | 0.50 | 0.02 | 0.04 | 48 |
| **macro avg** | 0.62 | 0.51 | 0.44 | 180 |
| **weighted avg** | 0.67 | 0.73 | 0.63 | 180 |

> ⚠️ Distress recall = 0.02 — model defaults to non_distress. 73% accuracy is misleading due to class imbalance. See postmortem.

### Multi-task (KWS + Distress)

| Task | Test Accuracy |
|---|---|
| KWS | 62.98% |
| Distress | 73.33% |

---

## ⚡ Model Efficiency (Latency + Size)

Measured on **Apple MPS**:

| Metric | Value |
|---|---|
| Forward pass (model-only) avg | **0.443 ms** |
| End-to-end (feature extraction + forward) avg | **1.17 ms** |
| **End-to-end p95** | **2.18 ms** |
| Parameters | 111,051 |
| State dict size | **0.43 MB** |

**Edge-deployable:** < 0.5 MB and sub-3ms p95 makes this suitable for on-device inference (mobile, embedded mic arrays).

---

## 🛠️ MLOps & Infrastructure

### Reproducibility
- All experiments tracked with fixed random seeds.
- Dataset split (train/val/test) is **fixed and version-controlled** to prevent leakage across variants.
- Feature extraction config (sample rate, n_mels, n_mfcc, frame length) stored in a central `config.py`.

### Experiment tracking
- Results table maintained manually across 4 variants (CLEAN-LOGMEL, NOISE-LOGMEL, SPECAUG-LOGMEL, MFCC-NOISE).
- Next step: integrate **MLflow** or **Weights & Biases** for automated metric logging per run.

### CI/CD awareness
- The pipeline is structured to support CI gates: model is promoted only if clean accuracy > 85% AND 0 dB SNR accuracy > 70%.
- Planned: GitHub Actions workflow to run evaluation on every PR merge, blocking regressions.

### Deployment path
```
Train (local / Colab)
        │
        ▼
Export → TorchScript / ONNX (0.43 MB)
        │
        ▼
Serve via FastAPI endpoint  ──► or ──► On-device (CoreML / TFLite)
        │
        ▼
Observability: log p95 latency per request, alert if > 5ms
```

### Monitoring & alerts (planned)
- Log per-request latency; alert if p95 drifts above 5 ms threshold.
- Track live false positive rate on non-distress traffic; alert if FPR > 5%.
- Shadow testing: new model variant runs alongside production, outputs compared before cutover.

---

## 🔥 Postmortem: What Broke and How We Fixed It

### Issue 1 — Distress classifier defaulted to majority class
**What happened:** The binary distress classifier achieved 73% accuracy but distress recall = 0.02. It was predicting `non_distress` for nearly all inputs.

**Root cause:** Severe class imbalance — 132 non-distress vs 48 distress samples (ratio ~2.75:1). The model minimized cross-entropy loss by collapsing to the majority class.

**Fix applied:** Identified the problem via confusion matrix (47/48 distress samples misclassified). Added weighted loss (`pos_weight` in `BCEWithLogitsLoss`) to penalize false negatives more heavily.

**Status:** Fix implemented; retraining with weighted loss + oversampling of distress class is in progress. Target: distress recall > 0.70.

---

### Issue 2 — SpecAugment degraded performance
**What happened:** SPECAUG-LOGMEL scored worst in both clean (78.57%) and noisy conditions (67.63% at 0 dB), underperforming even the clean-trained baseline.

**Root cause:** SpecAugment is most effective with large datasets (tens of thousands of diverse samples per class). With our small distress-class subset, masking too aggressively destroyed signal rather than building robustness.

**Fix applied:** Disabled SpecAugment for the primary model path. Narrowed augmentation to noise mixing only (MFCC-NOISE), which yielded the best results across all SNR conditions.

**Lesson:** Regularization strength must be calibrated to dataset size. More augmentation ≠ more robustness at small scale.

---

### Issue 3 — Multi-task learning hurt KWS accuracy
**What happened:** Multi-task model achieved only 62.98% KWS accuracy vs 88.37% for single-task MFCC-NOISE.

**Root cause:** Gradient conflict between KWS and distress heads — the distress task (tiny dataset) dominated gradient updates, pulling the shared backbone away from the KWS optimum.

**Fix applied:** Decoupled the two tasks into independent models for evaluation. Multi-task remains experimental; single-task MFCC-NOISE is the production candidate.

**Next steps:** Explore gradient surgery or task-specific learning rates to reduce interference.

---

## 🔁 Reliability: Caching, Fallbacks, Observability

| Concern | Approach |
|---|---|
| **Latency spikes** | Feature extraction + forward pass cached for identical audio chunks (hash-based) |
| **Model failures** | Fallback: if model throws, log error and return `distress_detected: false` with low-confidence flag |
| **Data drift** | Periodic evaluation on held-out real recordings; alert if accuracy drops > 3% month-over-month |
| **Eval gates** | Promote new model only if: clean acc > 85% AND 0 dB acc > 70% AND distress recall > 0.70 |
| **Rollback** | Model checkpoints versioned; rollback = swap checkpoint path in config |

---

## 🚀 Quickstart

```bash
git clone https://github.com/AKilalours/noise-robust-kws-distress-detection
cd noise-robust-kws-distress-detection

pip install -r requirements.txt

# Download Kaggle datasets and place under data/
# See Data Sources section for links

# Run full pipeline (preprocessing → training → evaluation)
jupyter notebook ASR.ipynb
```

---

## 📁 Repository Structure

```
noise-robust-kws-distress-detection/
├── ASR.ipynb                  # End-to-end pipeline: data prep → training → eval
├── fpr-Team-6-2.pdf           # Final report
├── requirements.txt
└── README.md
```

---

## 📄 Report

Full methodology, ablation analysis, and extended results: [`fpr-Team-6-2.pdf`](./fpr-Team-6-2.pdf)

---

## 👥 Team & Contributions

This is a collaborative team project — all work was split equally between both members.

| | **Akila Lourdes Miriyala Francis** | **Akilan Manivannan** |
|---|---|---|
| 🎙️ **Data & Preprocessing** | Audio collection & team voice recordings, silence trimming, resampling pipeline | Kaggle dataset sourcing, noise mixing implementation, SNR-level augmentation |
| 🧠 **Modelling** | Log-Mel spectrogram feature extraction, CLEAN-LOGMEL & NOISE-LOGMEL variants | MFCC feature pipeline, MFCC-NOISE & SPECAUG-LOGMEL variants |
| 🏗️ **Architecture** | KWS CNN backbone design, multi-task head integration | Distress binary classifier, loss weighting & class imbalance analysis |
| 📊 **Evaluation** | Noise robustness evaluation (SNR buckets), confusion matrix analysis | Precision/Recall/F1 reporting, per-class classification reports |
| ⚡ **Efficiency & Ops** | Model export, latency benchmarking (p95, forward pass timing) | Parameter counting, model size profiling, Apple MPS optimisation |
| 📝 **Documentation** | Architecture diagrams, postmortem write-up | Results tables, report compilation (`fpr-Team-6-2.pdf`) |

> **Equal contribution** — all design decisions, experiments, and findings were discussed and validated jointly.

---

<div align="center">
<sub>Built with PyTorch · MFCC · CNN · Noise Augmentation · Apple MPS</sub><br/>
<sub>Team 6 · Akila Lourdes Miriyala Francis & Akilan Manivannan</sub>
</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7a2e00,50:ff7b00,100:ffb347&height=120&section=footer" width="100%"/>
