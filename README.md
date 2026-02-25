# 🧬 BioEcho

**Multimodal AI Health Biomarker Prediction System**  
_AMD Slingshot 2026 — Submission_

BioEcho is a multimodal AI system that predicts 6 health biomarkers from a 60-second phone scan — using only your voice, camera, eye movement, and typing. No wearable. No blood test. No clinic.

It builds a personal "Bio Signature" — a 256-dimensional vector that fingerprints your unique health state. Tracked weekly, it detects health deterioration up to 6 months before clinical symptoms appear.

## ✨ System Architecture

BioEcho analyzes 4 input modalities, fuses them, and predicts 6 target biomarkers.

### 🎙️ 1. Voice

- **Input:** 60-second audio
- **Features:** HandCrafted (MFCC, Jitter, Shimmer, HNR, Spectral, F0, Tempo) & Deep (Wav2Vec2 768-dim contextual speech embedding)
- **Detects:** Parkinson's tremor, Depression, Diabetes, Stress

### 📸 2. Face (rPPG)

- **Input:** 60-second video from phone front camera
- **Features:** Heart Rate, HRV (RMSSD, SDNN, pNN50) via CHROM algorithm
- **Detects:** Cardiac stress, Anaemia, Blood pressure proxy

### 👁️ 3. Eye Tracking

- **Input:** Video via MediaPipe iris
- **Features:** Gaze position, velocity, saccade analysis, pupil diameter, blinks
- **Detects:** Cognitive load, ADHD (fixation), Alzheimer's latency

### ⌨️ 4. Typing Dynamics

- **Input:** Background keyboard logging during scan
- **Features:** Dwell time, flight time, digraph latency, error rate, rhythm
- **Detects:** Motor decline, Tremor, Stress

## 🧠 Model Pipeline

- **Fusion:** A `CrossModalTransformer` treats the 4 modality embeddings as a unified sequence with self-attention. It uses modal dropout (randomly dropping a modality during training for robustness).
- **Output Targets:**
  1. **Heart Rate:** beats per minute
  2. **HRV RMSSD:** heart rate variability (ms)
  3. **Stress Score:** 0-100 proxy score
  4. **Cognitive Load:** 0-100 mental workload index
  5. **Neurological Risk:** 0-100 neurological risk score
  6. **Bio Score:** 0-100 overall health trajectory (the main metric)

## 💻 Tech Stack & Repository Contents

The system architecture features `AudioEncoder`, `RPPGEncoder`, `GazeEncoder`, and `KeystrokeEncoder`, feeding into custom prediction and uncertainty heads (~15-20M parameters total).

- **`bioecho_master.py` & `bioecho_ui.py`:** Core source code for models, data processing, and user interface.
- **`bioecho_v2_fixed.ipynb` / `notebook_fixed.ipynb`:** Jupyter Notebooks containing the full Kaggle training pipeline, from global config down to early stopping, evaluation plots, synthetic data generation, and Mahalanobis drift detection.
- **`run_bioecho.bat`:** Windows batch script to launch the application components.

## 🚀 Key Technical Decisions

- **Gaussian NLL Loss:** Predicts a target value along with _uncertainty estimation_. Overconfident wrong predictions in healthcare are dangerous.
- **Modal Dropout:** Forces the fusion transformer to never rely 100% on a single signal (e.g., if the camera fails, it leans on voice and typing).
- **Mahalanobis Distance for Drift:** Outperforms simple threshold alerts by analyzing the variance in your unique multidimensional baseline.
- **Wav2Vec2 + Handcrafted Audio Features:** Gated fusion allows the model to learn when to trust the deep embedding vs engineered clinical features.

## 🛠️ AMD Ryzen AI Integration

BioEcho is specifically designed to leverage **AMD Ryzen AI NPUs**.

- **ONNX export:** Utilizes standard formats for AMD Ryzen AI SDK deployment.
- **On-device Privacy:** Removes cloud dependence, which is a key differentiator.
- **Performance:** Model execution scales down to ~1.1 ms on a Ryzen AI NPU compared to ~14.0 ms on standard Intel i7 CPUs, making real-time interactive feedback possible.

## 🤝 Community & Contact

**Built for the AMD Slingshot 2026 — Bengaluru Campus Day (27-28 April 2026)**
_Theme: AI for Social Good / Open Innovation_

---

> _"Your body has been sending warnings for months. BioEcho is the first system that listens."_
