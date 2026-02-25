# 🚀 BioEcho v2 — System Architecture & Training Guide

BioEcho is an advanced multimodal AI system designed to predict six key health biomarkers utilizing non-invasive data streams from a standard PC desktop or laptop. It computes a 256-dimensional "Bio Signature"—a real-time fingerprint of a user's physiological and neurological state.

This project is strictly committed to **Open Source** and is constrained to **Non-Commercial Research Use** only, due to the clinical datasets involved.

---

## 🧠 Core System Architecture

BioEcho captures four distinct non-invasive modalities simultaneously, processes them through specialized encoders, and fuses them to make holistic health predictions.

### 1. 🎙️ Voice Architecture (AudioEncoder)

- **Input Stream:** 60-second continuous audio recording via PC microphone.
- **Dual-Stream Processing:**
  1. **Acoustic Engine:** Uses OpenSMILE to extract deterministic, clinically validated handcrafted features (MFCCs, Jitter, Shimmer, HNR, Spectral Centroids).
  2. **Deep Semantic Engine:** Utilizes a pre-trained `Wav2Vec2` transformer base (fine-tuned with LoRA) to extract 768-dimensional contextual speech embeddings.
- **Biomarker Targets:** Parkinson's tremor detection, depression voice markers, and acute stress.

### 2. 📸 Face & Vitals Architecture (RPPGEncoder)

- **Input Stream:** 60-second video stream from a standard PC webcam.
- **rPPG (Remote Photoplethysmography):** The system isolates Regions of Interest (ROIs) on the face (forehead, cheeks) and tracks micro-variations in skin color caused by capillary blood flow during each cardiac cycle.
- **Processing:** A 3-branch multi-scale 1D CNN analyzes the CHROM signal to extract Heart Rate (bpm) and Heart Rate Variability (HRV - specifically RMSSD and pNN50).
- **Biomarker Targets:** Cardiac stress, resting heart rate, and cardiovascular efficiency.

### 3. 👁️ Gaze & Neurological Architecture (GazeEncoder)

- **Input Stream:** Gaze tracking derived from the PC webcam via MediaPipe Iris tracking.
- **Processing:** A Temporal Attention-based Bidirectional LSTM tracks saccadic velocity (the speed of eye movements between fixation points), pupil dilation variability, and blink rates.
- **Biomarker Targets:** High correlations with Cognitive Load (pupil dilation) and early-stage neurological decline (delayed saccade latency).

### 4. ⌨️ Keystroke Dynamics (KeystrokeEncoder)

- **Input Stream:** Background OS-level keyboard hook tracking rhythm during the 60-second session.
- **Processing:** A custom Transformer Encoder maps flight times (key-up to key-down) and dwell times (key-down to key-up) into a continuous sequence vector.
- **Biomarker Targets:** Motor skill degradation (tremors) and acute mental fatigue.

---

## 🧬 The Fusion Engine: CrossModalTransformer

The real power of BioEcho is the fusion mechanism.
All four encoded modalities (Voice, Face, Gaze, Typing) are projected into separate 256-dimensional vectors. They are then fed into a **CrossModalTransformer** along with a special `[FUSION]` classification token.

- **Self-Attention Mechanism:** The transformer allows each modality to contextually attend to the others. For example, if Voice indicates high stress, the model cross-references the Face HRV data to confirm if physiological stress matches the vocal stress.
- **Modal Dropout (Crucial for Edge Deployments):** During training, modalities are randomly zeroed out at a 15% rate. This ensures the transformer does not over-rely on a single sensor (e.g., if the user is in a dark room and the webcam fails, the system seamlessly falls back to Voice and Keystroke data without crashing).
- **Gaussian NLL Loss:** The prediction heads output the biomarker value _and_ a predicted variance (uncertainty score). In healthcare, knowing when an AI is uncertain is just as important as the prediction itself.

---

## 🏋️ The 6-Stage Kaggle Training Protocol

To keep the project accessible and 100% free, the entire 15+ million parameter model can be trained from scratch on free Kaggle kernels utilizing Dual T4 GPUs.

The training protocol is split into 6 isolated Jupyter Notebooks (located in `training/files/`).

1. **`1 voice.ipynb`**: Trains the Wav2Vec2 + LoRA stack on CREMA-D and MDVR-KCL datasets. Exports `voice_int8.onnx`.
2. **`2_face.ipynb`**: Trains the 3D CNN PhysNet on UBFC-rPPG for Blood Pressure & HRV. Exports `rppg_int8.onnx`.
3. **`3_gaze.ipynb`**: Trains the Multi-scale BiLSTM on the MPIIGaze dataset. Exports `gaze_int8.onnx` and `saccade_int8.onnx`.
4. **`4_typing.ipynb`**: Trains the Sequence Transformer on background keystroke data. Exports `key_int8.onnx`.
5. **`5_face_disease.ipynb`**: Trains an EfficientNet-V2-S for skin-level lesion and jaundice detection on HAM10000. Exports `face_int8.onnx`.
6. **`6_fusion.ipynb`**: The master module. It freezes the encoder weights and trains the `CrossModalTransformer` to map the combinations to the clinical targets. Exports `bioecho_fusion_int8.onnx`.

### ⚡ Optimization & Deployment (AMD Ryzen AI)

Once trained, the PyTorch models are quantized down to **INT8 ONNX** formats. This reduces memory footprint by 4x and optimizes them specifically to run at extreme speeds on Edge NPUs, such as the **AMD Ryzen AI NPU architecture**, providing sub-second real-time inference without requiring a cloud server.
