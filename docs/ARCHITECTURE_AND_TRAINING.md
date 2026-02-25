# 🚀 BioEcho v2 — How It Works & Training Guide

BioEcho is a multimodal AI system that predicts health biomarkers from a 60-second phone scan (using voice, camera, eye movement, and typing). It builds a 256-dimensional "Bio Signature" to fingerprint your health state. This project is committed to being **100% Open Source and Free**.

## 🧠 How It Works

BioEcho captures 4 non-invasive modalities simultaneously:

1. **🎙️ Voice (AudioEncoder):**
   - Takes 60 seconds of audio.
   - Extracts handcrafted features (OpenSMILE) and deep contextual features (Wav2Vec2 + LoRA).
   - Detects Parkinson's tremor, depression markers, and stress.

2. **📸 Face (RPPGEncoder):**
   - Uses the phone's front camera.
   - Performs rPPG (Remote Photoplethysmography) to extract Heart Rate and Heart Rate Variability (HRV) from subtle skin color changes.

3. **👁️ Gaze (GazeEncoder & Saccade Analysis):**
   - Tracks eye movement, saccade velocity, and blink rates via MediaPipe.

4. **⌨️ Keystroke Dynamics (KeystrokeEncoder):**
   - Analyzes typing rhythm, flight time, and dwell time in the background to assess cognitive/motor load.

**Fusion Pipeline:**
All 4 modalities are passed through a `CrossModalTransformer` for fusion. It uses "Modal Dropout" to make the model robust even if one sensor fails (e.g., in loud environments). The final output is predicting 6 targets like Stress Score, Cognitive Load, and Neurological Risk.

---

## 🏋️ How We Are Going to Train the Model

The model is structured for Kaggle Dual T4 GPU training to keep infrastructure costs at zero for open-source contributors. We split the training into 6 specialized Jupyter Notebooks located in `training/files/`.

### ⚡ General Kaggle Optimizations

- **Environment:** DataParallel 2× T4 GPUs with FP16 precision.
- **Speed:** Heavily optimized epochs and batch sizes (e.g., 10 epochs for Voice, 15 for rPPG).
- **Early Stopping:** Captures peak performance without overfitting.
- **Quantization:** Final models are exported to `INT8 ONNX` formats to run efficiently on CPUs or edge NPUs inside `bioecho_ui.py`.

### 📓 The 6-Stage Training Protocol

**1. Voice Biomarker (`1 voice.ipynb`)**

- **Goal:** Train Wav2Vec2 + LoRA.
- **Data:** CREMA-D (emotion) & MDVR-KCL (Parkinson's).
- **Exports:** `voice_int8.onnx`

**2. rPPG / Vital Signs (`2_face.ipynb`)**

- **Goal:** Train PhysNet 3D CNN for Blood Pressure & HRV.
- **Data:** UBFC-rPPG & MMPD-rPPG.
- **Exports:** `rppg_int8.onnx`

**3. Gaze & Eye Tracking (`3_gaze.ipynb`)**

- **Goal:** Train an iTracker + Multi-scale BiLSTM.
- **Data:** MPIIGaze.
- **Exports:** `gaze_int8.onnx` & `saccade_int8.onnx`

**4. Keystroke Dynamics (`4_typing.ipynb`)**

- **Goal:** Train a custom Transformer for sequencing keystrokes.
- **Data:** Synthetic sequences generated.
- **Exports:** `key_int8.onnx`

**5. Face Disease (`5_face_disease.ipynb`)**

- **Goal:** Train EfficientNet-V2-S for anemia, jaundice, and skin lesions.
- **Data:** HAM10000 & CelebA.
- **Exports:** `face_int8.onnx`

**6. Fusion Model (`6_fusion.ipynb`)**

- **Goal:** Combine all trained embeddings using Pairwise Cross-Attention.
- **Exports:** `bioecho_fusion_int8.onnx` (the final decision model).

### 🚀 Running the App locally

Once you have your `INT8 ONNX` models extracted from Kaggle (make sure `voice_int8.onnx` is located at `training/voice/voice_int8.onnx`), you can start the application:

```bash
run_bioecho.bat
# or
python bioecho_ui.py
```
