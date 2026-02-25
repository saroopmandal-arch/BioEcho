"""
BioEcho — Multimodal AI Health Biomarker Desktop Interface
Real-time voice analysis with camera face detection.
"""

import os
import sys
import time
import threading
import queue
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import sounddevice as sd
from scipy.io import wavfile
import onnxruntime as ort

# ── Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = SCRIPT_DIR / "training" / "voice" / "voice_int8.onnx"
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Audio config (must match ONNX model input shape)
SAMPLE_RATE = 16000
MAX_DURATION = 11.025  # 176400 samples / 16000 Hz
MAX_SAMPLES = 176400   # exact ONNX model input shape
SMILE_DIM = 6373

# ── Emotion labels for display
BINARY_TASKS = ["parkinsons", "depression", "respiratory"]
REGRESS_TASKS = ["emotion_valence", "emotion_arousal",
                 "bp_systolic", "hrv_sdnn", "cognitive_load"]

# ── Color palette
BG_DARK = "#08080f"
BG_PANEL = "#10101c"
BG_CARD = "#181830"
BG_CARD_HOVER = "#1e1e3a"
ACCENT = "#00e5b0"
ACCENT_DIM = "#007a63"
ACCENT_GLOW = "#00ffcc"
DANGER = "#ff4757"
WARNING = "#ffa502"
TEXT_PRIMARY = "#eaeaf4"
TEXT_SECONDARY = "#7777a0"
SUCCESS = "#2ed573"
BORDER_SUBTLE = "#252540"


def get_microphone_list():
    """Get list of available input audio devices from the default Host API."""
    mic_list = []
    try:
        hostapi = sd.default.hostapi
        devices = sd.query_devices()
        
        for i, d in enumerate(devices):
            # Only use devices from the default stable API (prevents silence bugs)
            if d["max_input_channels"] > 0 and d["hostapi"] == hostapi:
                name = d["name"]
                # Clean up name (remove API suffix if any)
                if len(name) > 42:
                    name = name[:39] + "..."
                mic_list.append((i, name))
                
        # If somehow empty, fallback to default device
        if not mic_list:
            default_in = sd.default.device[0]
            if default_in >= 0:
                mic_list.append((default_in, devices[default_in]["name"]))
    except Exception as e:
        print(f"[Audio Warning] {e}")
        
    return mic_list


class AudioRecorder:
    """Thread-safe audio recorder using sounddevice."""

    def __init__(self, sample_rate=SAMPLE_RATE, device_index=None):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.recording = False
        self.audio_buffer = []
        self.stream = None
        self.lock = threading.Lock()

    def set_device(self, device_index):
        """Change the recording device."""
        self.device_index = device_index

    def _callback(self, indata, frames, time_info, status):
        """Called for each audio block."""
        if self.recording:
            with self.lock:
                self.audio_buffer.append(indata[:, 0].copy())

    def start(self):
        """Start recording."""
        with self.lock:
            self.audio_buffer = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
            device=self.device_index
        )
        self.stream.start()

    def stop(self):
        """Stop recording and return the audio."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.lock:
            if self.audio_buffer:
                audio = np.concatenate(self.audio_buffer)
            else:
                audio = np.zeros(1, dtype=np.float32)
            self.audio_buffer = []
        return audio

    def get_current_amplitude(self):
        """Get the current amplitude for waveform display."""
        with self.lock:
            if self.audio_buffer:
                recent = self.audio_buffer[-1] if self.audio_buffer else np.zeros(1)
                return float(np.abs(recent).mean())
        return 0.0


class VoiceInference:
    """Runs ONNX inference on recorded audio."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.session = None
        self.loaded = False
        self.error_msg = ""

    def load(self):
        """Load ONNX model."""
        try:
            providers = ort.get_available_providers()
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.loaded = True
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            print(f"[Model Loaded] Inputs: {[(i.name, i.shape) for i in inputs]}")
            print(f"[Model Loaded] Outputs: {[(o.name, o.shape) for o in outputs]}")
            return True
        except Exception as e:
            self.error_msg = str(e)
            print(f"[Model Error] {e}")
            return False

    def preprocess_audio(self, audio):
        """Preprocess audio to match training pipeline."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            audio = audio / (rms + 1e-8) * 0.1
        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        elif len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        return audio.astype(np.float32)

    def extract_features(self, audio):
        """Extract OpenSMILE ComParE_2016 features — same as training pipeline."""
        try:
            import opensmile
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            feats = smile.process_signal(audio, SAMPLE_RATE)
            feat_vector = feats.values.flatten().astype(np.float32)

            # Pad or trim to SMILE_DIM (6373)
            smile_features = np.zeros(SMILE_DIM, dtype=np.float32)
            n = min(len(feat_vector), SMILE_DIM)
            smile_features[:n] = feat_vector[:n]
            return smile_features
        except Exception as e:
            print(f"[OpenSMILE Error] {e}, falling back to zeros")
            return np.zeros(SMILE_DIM, dtype=np.float32)

    def run(self, audio):
        """Run inference on audio, returns results dict."""
        if not self.loaded:
            return None

        processed_audio = self.preprocess_audio(audio)
        smile_features = self.extract_features(audio)
        


        audio_input = processed_audio[np.newaxis, :]
        smile_input = smile_features[np.newaxis, :]

        try:
            outputs = self.session.run(None, {
                "audio": audio_input,
                "smile_features": smile_input
            })
            voice_embedding = outputs[0][0]
            binary_risks = outputs[1][0]
            regression_scores = outputs[2][0]
            

            binary_probs = 1 / (1 + np.exp(-binary_risks))

            results = {
                "parkinsons_risk": float(binary_probs[0]) * 100,
                "depression_risk": float(binary_probs[1]) * 100,
                "respiratory_risk": float(binary_probs[2]) * 100,
                "emotion_valence": float(regression_scores[0]),
                "emotion_arousal": float(regression_scores[1]),
                "bp_systolic": float(regression_scores[2]) * 200,
                "hrv_sdnn": float(regression_scores[3]) * 100,
                "cognitive_load": float(regression_scores[4]) * 100,
                "embedding": voice_embedding,
            }
            risk_avg = (results["parkinsons_risk"] + results["depression_risk"]
                        + results["respiratory_risk"]) / 3.0
            results["bio_score"] = max(0, min(100, 100 - risk_avg))
            results["stress_score"] = max(0, min(100,
                abs(results["emotion_arousal"]) * 50 + results["cognitive_load"] * 0.5))
            return results
        except Exception as e:
            print(f"[Inference Error] {e}")
            return None


def generate_report(results, audio_duration):
    """Generate a text report file from inference results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"bioecho_report_{timestamp}.txt"

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║           🧬 BioEcho — Health Biomarker Report              ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Audio   : {audio_duration:.1f} seconds recorded",
        f"  Model   : voice_int8.onnx (INT8 Quantized)",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  OVERALL BIO SCORE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  Bio Score       : {results['bio_score']:.1f} / 100",
        f"  Stress Score    : {results['stress_score']:.1f} / 100",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  CLINICAL RISK INDICATORS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  Parkinson's Risk  : {results['parkinsons_risk']:.1f}%"
        f"  {'⚠️ ELEVATED' if results['parkinsons_risk'] > 50 else '✅ Normal'}",
        f"  Depression Risk   : {results['depression_risk']:.1f}%"
        f"  {'⚠️ ELEVATED' if results['depression_risk'] > 50 else '✅ Normal'}",
        f"  Respiratory Risk  : {results['respiratory_risk']:.1f}%"
        f"  {'⚠️ ELEVATED' if results['respiratory_risk'] > 50 else '✅ Normal'}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  VITAL ESTIMATES",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  Blood Pressure (systolic) : {results['bp_systolic']:.0f} mmHg",
        f"  HRV (SDNN)               : {results['hrv_sdnn']:.1f} ms",
        f"  Cognitive Load            : {results['cognitive_load']:.1f}%",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  EMOTIONAL STATE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  Valence (pleasure)  : {results['emotion_valence']:+.2f}  "
        f"({'Positive' if results['emotion_valence'] > 0.2 else 'Negative' if results['emotion_valence'] < -0.2 else 'Neutral'})",
        f"  Arousal (energy)    : {results['emotion_arousal']:+.2f}  "
        f"({'High' if results['emotion_arousal'] > 0.3 else 'Low' if results['emotion_arousal'] < -0.3 else 'Medium'})",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  DISCLAIMER",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  This report is generated by an AI research prototype.",
        "  It is NOT a medical diagnosis. Always consult a qualified",
        "  healthcare professional for medical advice.",
        "",
        "  Model: BioEcho v2 — Voice Biomarker (RAVDESS/CREMA-D/MDVR-KCL)",
        "  ═══════════════════════════════════════════════════════════",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


class WaveformCanvas(ctk.CTkCanvas):
    """Custom canvas for drawing audio waveform animation."""

    def __init__(self, master, width=300, height=80, **kwargs):
        super().__init__(master, width=width, height=height,
                         bg=BG_CARD, highlightthickness=0, **kwargs)
        self.w = width
        self.h = height
        self.amplitudes = [0.0] * 80
        self.draw_idle()

    def draw_idle(self):
        """Draw idle waveform with subtle center line."""
        self.delete("all")
        mid_y = self.h // 2
        for i in range(0, self.w, 8):
            self.create_line(i, mid_y, i + 4, mid_y,
                             fill=BORDER_SUBTLE, width=1)

    def update_waveform(self, amplitude):
        """Add a new amplitude and redraw with smooth bars."""
        self.amplitudes.append(min(amplitude * 6, 1.0))
        self.amplitudes = self.amplitudes[-80:]
        self.delete("all")

        mid_y = self.h // 2
        bar_width = self.w / 80

        for i, amp in enumerate(self.amplitudes):
            x = i * bar_width
            bar_h = max(1, amp * mid_y * 0.85)
            # Fade older bars
            alpha = i / 80.0
            if amp > 0.08:
                color = ACCENT
            elif amp > 0.02:
                color = ACCENT_DIM
            else:
                color = BORDER_SUBTLE
            self.create_rectangle(
                x + 1, mid_y - bar_h,
                x + bar_width - 1, mid_y + bar_h,
                fill=color, outline=""
            )


def resize_with_aspect_ratio(frame, target_w, target_h):
    """Resize frame to fit within target while keeping aspect ratio.
    Adds black letterbox bars as needed."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas and center the frame
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


class BioEchoApp(ctk.CTk):
    """Main BioEcho Desktop Application."""

    def __init__(self):
        super().__init__()

        # ── Window config
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("🧬 BioEcho — AI Health Biomarker Scanner")
        self.geometry("1340x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        # ── Thread-safe callback queue (avoids Tkinter threading crash)
        self._callback_queue = queue.Queue()

        # ── State
        self.scanning = False
        self.scan_start_time = 0
        self.scan_duration = 12  # seconds (model needs ~11s of audio)
        self.last_results = None
        self.last_audio_duration = 0.0
        self.camera_active = False
        self.face_cascade = None
        self.cap = None
        self.scan_count = 0

        # ── Microphone list
        self.mic_list = get_microphone_list()
        self.selected_mic_index = self.mic_list[0][0] if self.mic_list else None

        # ── Components
        self.recorder = AudioRecorder(device_index=self.selected_mic_index)
        self.inference = VoiceInference(ONNX_MODEL_PATH)

        # ── Build UI
        self._build_header()
        self._build_main_layout()
        self._build_status_bar()

        # ── Load model in background
        self._load_model_async()

        # ── Start camera
        self._start_camera()

        # ── Start polling the callback queue
        self._poll_queue()

        # ── Protocol for clean exit
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _poll_queue(self):
        """Poll the callback queue for thread-safe GUI updates."""
        try:
            while not self._callback_queue.empty():
                callback = self._callback_queue.get_nowait()
                callback()
        except Exception:
            pass
        self.after(100, self._poll_queue)

    def _build_header(self):
        """Build the top header bar."""
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=56)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # Logo + Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=20, pady=8)

        ctk.CTkLabel(title_frame, text="🧬", font=ctk.CTkFont(size=26)).pack(side="left")
        ctk.CTkLabel(title_frame, text="BioEcho",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=ACCENT).pack(side="left", padx=(8, 4))
        ctk.CTkLabel(title_frame, text="v2",
                     font=ctk.CTkFont(size=11),
                     text_color=TEXT_SECONDARY).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(title_frame, text="│",
                     font=ctk.CTkFont(size=14),
                     text_color=BORDER_SUBTLE).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(title_frame, text="AI Health Biomarker Scanner",
                     font=ctk.CTkFont(size=13),
                     text_color=TEXT_SECONDARY).pack(side="left")

        # Scan counter
        self.scan_counter_label = ctk.CTkLabel(
            header, text="Scans: 0",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY
        )
        self.scan_counter_label.pack(side="right", padx=(0, 20))

        # Model status indicator
        self.model_status_label = ctk.CTkLabel(
            header, text="⏳ Loading Model...",
            font=ctk.CTkFont(size=11),
            text_color=WARNING
        )
        self.model_status_label.pack(side="right", padx=(0, 16))

    def _build_main_layout(self):
        """Build the main 2-column layout."""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=(6, 6))
        main.grid_columnconfigure(0, weight=3, minsize=420)
        main.grid_columnconfigure(1, weight=4, minsize=500)
        main.grid_rowconfigure(0, weight=1)

        # ── Left: Camera + Audio Controls
        left = ctk.CTkFrame(main, fg_color=BG_PANEL, corner_radius=10,
                             border_width=1, border_color=BORDER_SUBTLE)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._build_left_panel(left)

        # ── Right: Results Dashboard
        right = ctk.CTkFrame(main, fg_color=BG_PANEL, corner_radius=10,
                              border_width=1, border_color=BORDER_SUBTLE)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        """Camera feed + audio controls.
        Pack order: controls FIRST (bottom), then camera gets remaining space.
        """
        # ── Audio Waveform + Controls (pack FIRST so they're always visible)
        audio_frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8,
                                    border_width=1, border_color=BORDER_SUBTLE)
        audio_frame.pack(fill="x", side="bottom", padx=14, pady=(2, 10))

        audio_header = ctk.CTkFrame(audio_frame, fg_color="transparent")
        audio_header.pack(fill="x", padx=10, pady=(10, 2))
        ctk.CTkLabel(audio_header, text="🎙️  Voice Input",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_PRIMARY).pack(side="left")
        self.audio_timer_label = ctk.CTkLabel(
            audio_header, text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY
        )
        self.audio_timer_label.pack(side="right")

        # Waveform
        self.waveform = WaveformCanvas(audio_frame, width=400, height=60)
        self.waveform.pack(fill="x", padx=10, pady=4)

        # Scan button
        self.scan_btn = ctk.CTkButton(
            audio_frame, text="▶  Start Scan",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT_DIM,
            text_color=BG_DARK, height=44,
            corner_radius=8,
            command=self._toggle_scan
        )
        self.scan_btn.pack(fill="x", padx=10, pady=(4, 4))

        # Progress bar
        self.scan_progress = ctk.CTkProgressBar(
            audio_frame, progress_color=ACCENT,
            fg_color=BG_DARK, height=3
        )
        self.scan_progress.pack(fill="x", padx=10, pady=(0, 8))
        self.scan_progress.set(0)

        # ── Microphone Selection (pack SECOND from bottom)
        mic_frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8,
                                  border_width=1, border_color=BORDER_SUBTLE)
        mic_frame.pack(fill="x", side="bottom", padx=14, pady=(2, 6))

        mic_header = ctk.CTkFrame(mic_frame, fg_color="transparent")
        mic_header.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(mic_header, text="🎤  Microphone",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_PRIMARY).pack(side="left")

        # Mic dropdown
        mic_names = [name for _, name in self.mic_list]
        default_mic = mic_names[0] if mic_names else "No microphone found"
        self.mic_dropdown = ctk.CTkOptionMenu(
            mic_frame,
            values=mic_names if mic_names else ["No microphone found"],
            command=self._on_mic_changed,
            font=ctk.CTkFont(size=11),
            fg_color=BG_DARK,
            button_color=ACCENT_DIM,
            button_hover_color=ACCENT,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BG_CARD_HOVER,
            dropdown_text_color=TEXT_PRIMARY,
            text_color=TEXT_PRIMARY,
            width=300, height=30,
            corner_radius=6
        )
        self.mic_dropdown.set(default_mic)
        self.mic_dropdown.pack(fill="x", padx=10, pady=(0, 8))

        # ── Camera section header (pack LAST = appears at top, fills remaining space)
        cam_header = ctk.CTkFrame(parent, fg_color="transparent")
        cam_header.pack(fill="x", padx=14, pady=(14, 4))
        ctk.CTkLabel(cam_header, text="📷  Camera Feed",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=TEXT_PRIMARY).pack(side="left")
        self.face_count_label = ctk.CTkLabel(cam_header, text="",
                                              font=ctk.CTkFont(size=11),
                                              text_color=TEXT_SECONDARY)
        self.face_count_label.pack(side="right")

        # Camera canvas — gets remaining space, aspect ratio preserved in _update_camera
        self.camera_canvas = ctk.CTkLabel(parent, text="Initializing camera...",
                                           fg_color="#000000", corner_radius=8,
                                           text_color=TEXT_SECONDARY)
        self.camera_canvas.pack(fill="both", expand=True, padx=14, pady=(4, 6))

    def _build_right_panel(self, parent):
        """Results dashboard."""
        # Section header
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", padx=14, pady=(14, 6))
        ctk.CTkLabel(header_frame, text="📊  Analysis Results",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=TEXT_PRIMARY).pack(side="left")

        self.report_btn = ctk.CTkButton(
            header_frame, text="📄 Save Report",
            font=ctk.CTkFont(size=11),
            fg_color=BG_CARD, hover_color=ACCENT_DIM,
            text_color=TEXT_SECONDARY, width=110, height=28,
            corner_radius=6, state="disabled",
            border_width=1, border_color=BORDER_SUBTLE,
            command=self._save_report
        )
        self.report_btn.pack(side="right")

        # Scrollable results area
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent",
                                         scrollbar_button_color=BORDER_SUBTLE,
                                         scrollbar_button_hover_color=ACCENT_DIM)
        scroll.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Bio Score (big card)
        self._create_big_score_card(scroll)

        # ── Clinical Risk section
        self._create_section_label(scroll, "⚕️  Clinical Risk Assessment")

        risk_row = ctk.CTkFrame(scroll, fg_color="transparent")
        risk_row.pack(fill="x", padx=6, pady=2)
        risk_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.risk_cards = {}
        risk_data = [
            ("parkinsons_risk", "Parkinson's", "🧠"),
            ("depression_risk", "Depression", "💭"),
            ("respiratory_risk", "Respiratory", "🫁")
        ]
        for i, (key, label, icon) in enumerate(risk_data):
            card = self._create_risk_card(risk_row, f"{icon} {label}")
            card.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
            self.risk_cards[key] = card

        # ── Vital Estimates section
        self._create_section_label(scroll, "💓  Vital Estimates")

        vitals_row = ctk.CTkFrame(scroll, fg_color="transparent")
        vitals_row.pack(fill="x", padx=6, pady=2)
        vitals_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.vital_cards = {}
        vital_data = [
            ("bp_systolic", "Blood Pressure", "mmHg"),
            ("hrv_sdnn", "HRV (SDNN)", "ms"),
            ("cognitive_load", "Cognitive Load", "%")
        ]
        for i, (key, label, unit) in enumerate(vital_data):
            card = self._create_vital_card(vitals_row, label, unit)
            card.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
            self.vital_cards[key] = card

        # ── Emotional State section
        self._create_section_label(scroll, "😊  Emotional State")

        emo_row = ctk.CTkFrame(scroll, fg_color="transparent")
        emo_row.pack(fill="x", padx=6, pady=2)
        emo_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.emo_cards = {}
        emo_data = [
            ("emotion_valence", "Valence"),
            ("emotion_arousal", "Arousal"),
            ("stress_score", "Stress")
        ]
        for i, (key, label) in enumerate(emo_data):
            card = self._create_vital_card(emo_row, label, "")
            card.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
            self.emo_cards[key] = card

    def _create_section_label(self, parent, text):
        """Create a section header label."""
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_SECONDARY).pack(anchor="w", padx=14, pady=(10, 2))

    def _create_big_score_card(self, parent):
        """Create the big bio score card at the top."""
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10,
                              height=110, border_width=1, border_color=BORDER_SUBTLE)
        frame.pack(fill="x", padx=6, pady=(0, 4))
        frame.pack_propagate(False)

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(expand=True)

        ctk.CTkLabel(inner, text="OVERALL BIO SCORE",
                     font=ctk.CTkFont(size=10),
                     text_color=TEXT_SECONDARY).pack(pady=(2, 0))

        self.bio_score_value = ctk.CTkLabel(
            inner, text="—",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color=ACCENT
        )
        self.bio_score_value.pack()

        self.bio_score_bar = ctk.CTkProgressBar(
            frame, progress_color=ACCENT, fg_color=BG_DARK, height=5, width=320
        )
        self.bio_score_bar.pack(pady=(0, 10))
        self.bio_score_bar.set(0)

    def _create_risk_card(self, parent, label):
        """Create a risk indicator card."""
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8,
                              height=100, border_width=1, border_color=BORDER_SUBTLE)
        frame.pack_propagate(False)

        ctk.CTkLabel(frame, text=label,
                     font=ctk.CTkFont(size=11),
                     text_color=TEXT_SECONDARY).pack(pady=(12, 2))

        value_label = ctk.CTkLabel(frame, text="—",
                                    font=ctk.CTkFont(size=24, weight="bold"),
                                    text_color=TEXT_PRIMARY)
        value_label.pack()

        status_label = ctk.CTkLabel(frame, text="Awaiting scan",
                                     font=ctk.CTkFont(size=10),
                                     text_color=TEXT_SECONDARY)
        status_label.pack(pady=(0, 4))

        frame._value_label = value_label
        frame._status_label = status_label
        return frame

    def _create_vital_card(self, parent, label, unit):
        """Create a vital sign card."""
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8,
                              height=85, border_width=1, border_color=BORDER_SUBTLE)
        frame.pack_propagate(False)

        ctk.CTkLabel(frame, text=label,
                     font=ctk.CTkFont(size=11),
                     text_color=TEXT_SECONDARY).pack(pady=(10, 0))

        value_label = ctk.CTkLabel(frame, text="—",
                                    font=ctk.CTkFont(size=22, weight="bold"),
                                    text_color=TEXT_PRIMARY)
        value_label.pack()

        if unit:
            ctk.CTkLabel(frame, text=unit,
                         font=ctk.CTkFont(size=9),
                         text_color=TEXT_SECONDARY).pack()

        frame._value_label = value_label
        return frame

    def _build_status_bar(self):
        """Build the bottom status bar."""
        status = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=28)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            status, text="Ready — Click 'Start Scan' to begin",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY
        )
        self.status_label.pack(side="left", padx=14, pady=3)

        self.time_label = ctk.CTkLabel(
            status,
            text=datetime.now().strftime("%H:%M:%S"),
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY
        )
        self.time_label.pack(side="right", padx=14, pady=3)
        self._update_clock()

    # ── Microphone ──

    def _on_mic_changed(self, selected_name):
        """Handle microphone selection change."""
        for idx, name in self.mic_list:
            if name == selected_name:
                self.selected_mic_index = idx
                self.recorder.set_device(idx)
                self.status_label.configure(
                    text=f"🎤 Microphone: {selected_name}",
                    text_color=ACCENT)
                break

    # ── Model Loading ──

    def _load_model_async(self):
        """Load model in a background thread."""
        def _load():
            if ONNX_MODEL_PATH.exists():
                success = self.inference.load()
                self._callback_queue.put(lambda: self._on_model_loaded(success))
            else:
                self._callback_queue.put(lambda: self._on_model_loaded(False))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self, success):
        """Callback when model is loaded."""
        if success:
            self.model_status_label.configure(
                text="✅ Model Ready", text_color=SUCCESS)
            self.status_label.configure(
                text="Model loaded — Ready to scan")
        else:
            msg = "Model not found" if not ONNX_MODEL_PATH.exists() else self.inference.error_msg
            self.model_status_label.configure(
                text=f"❌ {msg[:40]}", text_color=DANGER)
            self.status_label.configure(
                text=f"Error: {msg}")

    # ── Camera ──

    def _start_camera(self):
        """Initialize and start camera feed."""
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.camera_canvas.configure(text="No camera detected")
                return

            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Load face cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            self.camera_active = True
            self._update_camera()
        except Exception as e:
            self.camera_canvas.configure(text=f"Camera error: {e}")

    def _update_camera(self):
        """Update camera frame at ~30 FPS with proper aspect ratio."""
        if not self.camera_active or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror

            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                color = (0, 229, 176)  # ACCENT in BGR
                thickness = 2
                corner_len = int(min(w, h) * 0.18)

                # Corner-style bounding box
                # Top-left
                cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
                cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
                # Top-right
                cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
                cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
                # Bottom-left
                cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
                cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
                # Bottom-right
                cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

                # Thin connecting lines between corners (subtle)
                thin = 1
                cv2.line(frame, (x + corner_len, y), (x + w - corner_len, y), color, thin)
                cv2.line(frame, (x + corner_len, y + h), (x + w - corner_len, y + h), color, thin)
                cv2.line(frame, (x, y + corner_len), (x, y + h - corner_len), color, thin)
                cv2.line(frame, (x + w, y + corner_len), (x + w, y + h - corner_len), color, thin)

                # Label with background
                label = "FACE DETECTED"
                if self.scanning:
                    label = "● SCANNING"
                    color = (0, 255, 100)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y - 2), (0, 0, 0), -1)
                cv2.putText(frame, label, (x + 3, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Convert to Tk image with PROPER aspect ratio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            canvas_w = self.camera_canvas.winfo_width()
            canvas_h = self.camera_canvas.winfo_height()
            if canvas_w > 20 and canvas_h > 20:
                frame_rgb = resize_with_aspect_ratio(frame_rgb, canvas_w, canvas_h)

            img = Image.fromarray(frame_rgb)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img,
                                  size=(canvas_w if canvas_w > 20 else 400,
                                        canvas_h if canvas_h > 20 else 300))
            self.camera_canvas.configure(image=imgtk, text="")
            self.camera_canvas._imgtk = imgtk

            n = len(faces)
            self.face_count_label.configure(
                text=f"{'🟢' if n > 0 else '⚫'} {n} face{'s' if n != 1 else ''} detected",
                text_color=ACCENT if n > 0 else TEXT_SECONDARY
            )

        self.after(33, self._update_camera)

    # ── Scanning ──

    def _toggle_scan(self):
        """Start or stop scanning."""
        if not self.inference.loaded:
            self.status_label.configure(text="⚠️ Model not loaded yet",
                                         text_color=WARNING)
            return
        if self.scanning:
            self._stop_scan()
        else:
            self._start_scan()

    def _start_scan(self):
        """Begin audio recording and scan."""
        self.scanning = True
        self.scan_start_time = time.time()
        self.scan_btn.configure(text="⏹  Stop Scan",
                                 fg_color=DANGER, hover_color="#cc3a47")
        self.status_label.configure(text="🔴  Recording voice — speak normally",
                                     text_color=DANGER)
        self.recorder.start()
        self._update_scan()

    def _stop_scan(self):
        """Stop recording and run inference."""
        self.scanning = False
        audio = self.recorder.stop()
        self.scan_btn.configure(text="⏳  Analyzing...",
                                 fg_color=WARNING, hover_color=WARNING,
                                 state="disabled")
        self.status_label.configure(text="🔄  Running AI inference on voice data...",
                                     text_color=WARNING)

        def _run():
            audio_duration = len(audio) / SAMPLE_RATE
            results = self.inference.run(audio)
            self._callback_queue.put(lambda r=results, d=audio_duration: self._on_results(r, d))

        threading.Thread(target=_run, daemon=True).start()

    def _update_scan(self):
        """Update scan progress and waveform."""
        if not self.scanning:
            return

        elapsed = time.time() - self.scan_start_time
        progress = min(elapsed / self.scan_duration, 1.0)
        remaining = max(0, self.scan_duration - elapsed)

        self.scan_progress.set(progress)
        self.audio_timer_label.configure(text=f"{remaining:.1f}s left")

        amp = self.recorder.get_current_amplitude()
        self.waveform.update_waveform(amp)

        if elapsed >= self.scan_duration:
            self._stop_scan()
        else:
            self.after(50, self._update_scan)

    def _on_results(self, results, audio_duration):
        """Display inference results."""
        self.scan_btn.configure(text="▶  Start Scan",
                                 fg_color=ACCENT, hover_color=ACCENT_DIM,
                                 state="normal")
        self.scan_progress.set(0)
        self.audio_timer_label.configure(text="Ready")
        self.waveform.draw_idle()

        if results is None:
            self.status_label.configure(text="❌ Inference failed",
                                         text_color=DANGER)
            return

        self.last_results = results
        self.last_audio_duration = audio_duration
        self.scan_count += 1
        self.scan_counter_label.configure(text=f"Scans: {self.scan_count}")

        # Update Bio Score
        score = results["bio_score"]
        score_color = SUCCESS if score >= 70 else WARNING if score >= 40 else DANGER
        self.bio_score_value.configure(text=f"{score:.0f}", text_color=score_color)
        self.bio_score_bar.configure(progress_color=score_color)
        self.bio_score_bar.set(score / 100)

        # Update risk cards
        for key, card in self.risk_cards.items():
            val = results[key]
            card._value_label.configure(text=f"{val:.1f}%")
            if val > 50:
                card._value_label.configure(text_color=DANGER)
                card._status_label.configure(text="⚠️ Elevated", text_color=DANGER)
            elif val > 25:
                card._value_label.configure(text_color=WARNING)
                card._status_label.configure(text="⚡ Monitor", text_color=WARNING)
            else:
                card._value_label.configure(text_color=SUCCESS)
                card._status_label.configure(text="✅ Normal", text_color=SUCCESS)

        # Update vital cards
        formats = {"bp_systolic": "{:.0f}", "hrv_sdnn": "{:.1f}",
                    "cognitive_load": "{:.1f}"}
        for key, card in self.vital_cards.items():
            fmt = formats.get(key, "{:.1f}")
            card._value_label.configure(text=fmt.format(results[key]))

        # Update emotion cards
        emo_fmts = {"emotion_valence": "{:+.2f}", "emotion_arousal": "{:+.2f}",
                     "stress_score": "{:.1f}"}
        for key, card in self.emo_cards.items():
            fmt = emo_fmts.get(key, "{:.1f}")
            card._value_label.configure(text=fmt.format(results[key]))

        # Enable report button
        self.report_btn.configure(state="normal", text_color=ACCENT)

        self.status_label.configure(
            text=f"✅  Scan #{self.scan_count} complete — Bio Score: {score:.0f}/100  |  "
                 f"Audio: {audio_duration:.1f}s",
            text_color=SUCCESS
        )

    # ── Report ──

    def _save_report(self):
        """Save the current results as a report."""
        if not self.last_results:
            return
        report_path = generate_report(self.last_results, self.last_audio_duration)
        self.status_label.configure(
            text=f"📄  Report saved: {report_path.name}",
            text_color=ACCENT
        )
        os.startfile(str(report_path))

    # ── Utilities ──

    def _update_clock(self):
        """Update the clock in the status bar."""
        self.time_label.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    def _on_close(self):
        """Clean shutdown."""
        self.camera_active = False
        if self.scanning:
            self.recorder.stop()
        if self.cap:
            self.cap.release()
        self.destroy()


def main():
    """Entry point."""
    print("=" * 60)
    print("  🧬 BioEcho — AI Health Biomarker Scanner v2")
    print("=" * 60)
    print(f"  Model  : {ONNX_MODEL_PATH}")
    print(f"  Exists : {ONNX_MODEL_PATH.exists()}")
    print(f"  Reports: {REPORTS_DIR}")

    mics = get_microphone_list()
    print(f"  Mics   : {len(mics)} found")
    for idx, name in mics:
        print(f"    [{idx}] {name}")
    print()

    app = BioEchoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
