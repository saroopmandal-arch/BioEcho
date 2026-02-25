# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 0 — Environment Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 1 — Global Config (BioEchoConfig dataclass)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BioEchoConfig:
    # Model dims
    embed_dim: int = 256
    audio_deep_dim: int = 768
    audio_handcrafted_dim: int = 200
    rppg_hr_features_dim: int = 4
    gaze_seq_len: int = 150
    gaze_feat_dim: int = 6 # e.g. x, y, velocity_x, velocity_y, pupil, blink
    keystroke_seq_len: int = 200
    keystroke_feat_dim: int = 5
    
    # Training
    batch_size: int = 16
    epochs: int = 50
    base_lr: float = 2e-4
    modal_dropout_rate: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = BioEchoConfig()

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 2 — Audio Preprocessing & Encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioEncoder(nn.Module):
    def __init__(self, cfg: BioEchoConfig):
        super().__init__()
        # MLP for handcrafted features (MFCC, Jitter, etc.)
        self.handcrafted_mlp = nn.Sequential(
            nn.Linear(cfg.audio_handcrafted_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg.embed_dim)
        )
        # Linear projection for deep features (Wav2Vec2)
        self.deep_proj = nn.Sequential(
            nn.Linear(cfg.audio_deep_dim, cfg.embed_dim),
            nn.Dropout(0.2)
        )
        # Learned fusion gate (softmax over two sources)
        self.gate = nn.Linear(cfg.embed_dim * 2, 2)
        
    def forward(self, handcrafted_x, deep_x):
        h1 = self.handcrafted_mlp(handcrafted_x) # (B, 256)
        h2 = self.deep_proj(deep_x)              # (B, 256)
        
        # Calculate attention weights over the two sources
        gates = F.softmax(self.gate(torch.cat([h1, h2], dim=-1)), dim=-1) # (B, 2)
        
        # Weighted sum: out = w1*h1 + w2*h2
        out = (h1 * gates[:, 0:1]) + (h2 * gates[:, 1:2])
        return out

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 3 — Face rPPG Pipeline & Encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RPPGEncoder(nn.Module):
    def __init__(self, cfg: BioEchoConfig, signal_length=300):
        super().__init__()
        # 3-branch multi-scale 1D CNN
        self.branch1 = self._make_branch(kernel_size=3)
        self.branch2 = self._make_branch(kernel_size=7)
        self.branch3 = self._make_branch(kernel_size=15)
        
        # Output of branches concatenated + linear projection
        self.signal_proj = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.GELU()
        )
        
        # Separate encoder for manual HR/HRV features (4-dim)
        self.hrv_proj = nn.Sequential(
            nn.Linear(cfg.rppg_hr_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256)
        )
        
        # Gated fusion
        self.gate = nn.Linear(256 * 2, 2)
        
    def _make_branch(self, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(150),
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1) # Global structure
        )

    def forward(self, rppg_signal, hrv_features):
        # rppg_signal: (B, 1, L)
        b1 = self.branch1(rppg_signal).squeeze(-1) # (B, 64)
        b2 = self.branch2(rppg_signal).squeeze(-1) # (B, 64)
        b3 = self.branch3(rppg_signal).squeeze(-1) # (B, 64)
        
        sig_feat = self.signal_proj(torch.cat([b1, b2, b3], dim=-1)) # (B, 256)
        hrv_feat = self.hrv_proj(hrv_features)                       # (B, 256)
        
        gates = F.softmax(self.gate(torch.cat([sig_feat, hrv_feat], dim=-1)), dim=-1)
        out = (sig_feat * gates[:, 0:1]) + (hrv_feat * gates[:, 1:2])
        return out

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 4 — Eye Tracking & Saccade Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GazeEncoder(nn.Module):
    def __init__(self, cfg: BioEchoConfig):
        super().__init__()
        # Bidirectional LSTM, hidden=128 -> 256 with bidir
        self.lstm = nn.LSTM(input_size=cfg.gaze_feat_dim, hidden_size=128, 
                            num_layers=3, batch_first=True, bidirectional=True)
        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, gaze_seq):
        # gaze_seq: (B, SeqLen, FeatDim)
        lstm_out, _ = self.lstm(gaze_seq) # (B, T, 256)
        
        # Temporal attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # (B, T, 1)
        
        # Weighted context vector
        context = torch.sum(lstm_out * attn_weights, dim=1) # (B, 256)
        return context

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 5 — Typing Keystroke Dynamics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KeystrokeEncoder(nn.Module):
    def __init__(self, cfg: BioEchoConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.keystroke_feat_dim, cfg.embed_dim)
        
        # Max seq length supported
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.keystroke_seq_len + 1, cfg.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.embed_dim, nhead=4, 
                                                   dim_feedforward=512, activation="gelu", 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, keystroke_seq):
        # keystroke_seq: (B, T, 5)
        B, T, _ = keystroke_seq.shape
        x = self.proj(keystroke_seq) # (B, T, 256)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, 256)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :T+1, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Output [CLS] token
        return x[:, 0, :]

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 7 — Model Architecture (CrossModalTransformer & Heads)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CrossModalTransformer(nn.Module):
    def __init__(self, cfg: BioEchoConfig):
        super().__init__()
        self.cfg = cfg
        # 4 modality tokens + 1 fusion token -> 5 token seq
        self.fusion_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim))
        self.modal_type_embeds = nn.Parameter(torch.randn(1, 5, cfg.embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.embed_dim, nhead=8,
                                                   dim_feedforward=1024, norm_first=True, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, mod_embs, training=True):
        # mod_embs: list of (B, 256) tensors -> num_modalities = 4
        B = mod_embs[0].shape[0]
        
        # Modal dropout during training
        if training and self.cfg.modal_dropout_rate > 0:
            for i in range(len(mod_embs)):
                if torch.rand(1).item() < self.cfg.modal_dropout_rate:
                    mod_embs[i] = torch.zeros_like(mod_embs[i])
                    
        # Stack modalities: (B, 4, 256)
        x = torch.stack(mod_embs, dim=1)
        
        # Prepend fusion token: (B, 5, 256)
        fusion_tokens = self.fusion_token.expand(B, -1, -1)
        x = torch.cat([fusion_tokens, x], dim=1)
        
        # Add modal type embeddings
        x = x + self.modal_type_embeds
        
        # Apply transformer
        x = self.transformer(x)
        
        # Return [FUSION] token context
        return x[:, 0, :]

class BioEchoModel(nn.Module):
    def __init__(self, cfg: BioEchoConfig):
        super().__init__()
        self.audio_enc = AudioEncoder(cfg)
        self.rppg_enc = RPPGEncoder(cfg)
        self.gaze_enc = GazeEncoder(cfg)
        self.keystroke_enc = KeystrokeEncoder(cfg)
        self.fusion = CrossModalTransformer(cfg)
        
        self.target_names = ["heart_rate", "hrv_rmssd", "stress_score", 
                             "cognitive_load", "neuro_risk", "bio_score"]
                             
        # Prediction Heads (x6)
        self.pred_heads = nn.ModuleDict()
        self.unc_heads = nn.ModuleDict()
        
        for name in self.target_names:
            self.pred_heads[name] = nn.Sequential(
                nn.Linear(cfg.embed_dim, 128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
            )
            # Uncertainty predicting log variance or stdev
            self.unc_heads[name] = nn.Sequential(
                nn.Linear(cfg.embed_dim, 64), nn.GELU(),
                nn.Linear(64, 1), nn.Softplus() # Variance must be positive
            )
            
    def forward(self, audio_hc, audio_deep, rppg_sig, rppg_hrv, gaze_seq, keystroke_seq):
        a_emb = self.audio_enc(audio_hc, audio_deep)
        r_emb = self.rppg_enc(rppg_sig, rppg_hrv)
        g_emb = self.gaze_enc(gaze_seq)
        k_emb = self.keystroke_enc(keystroke_seq)
        
        bio_sig = self.fusion([a_emb, r_emb, g_emb, k_emb], training=self.training)
        
        preds = {}
        uncs = {}
        for name in self.target_names:
            preds[name] = self.pred_heads[name](bio_sig)
            uncs[name] = self.unc_heads[name](bio_sig) + 1e-6 # numerical stability
            
        return bio_sig, preds, uncs

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 6 — Synthetic Data Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BioEchoDataset(Dataset):
    def __init__(self, num_samples, cfg: BioEchoConfig):
        self.cfg = cfg
        self.num_samples = num_samples
        
        # Modality 1: Audio
        self.audio_hc = torch.randn(num_samples, cfg.audio_handcrafted_dim)
        self.audio_deep = torch.randn(num_samples, cfg.audio_deep_dim)
        
        # Modality 2: rPPG
        self.rppg_sig = torch.randn(num_samples, 1, 300)
        self.rppg_hrv = torch.randn(num_samples, cfg.rppg_hr_features_dim)
        
        # Modality 3: Gaze
        self.gaze_seq = torch.randn(num_samples, cfg.gaze_seq_len, cfg.gaze_feat_dim)
        
        # Modality 4: Keystrokes
        self.keystroke_seq = torch.randn(num_samples, cfg.keystroke_seq_len, cfg.keystroke_feat_dim)
        
        # Targets (Realistic Synthetic Scales)
        self.targets = {
            "heart_rate": torch.randn(num_samples, 1) * 15 + 75.0, # 60-90 bpm
            "hrv_rmssd": torch.randn(num_samples, 1) * 20 + 40.0,  # RMSSD ms
            "stress_score": torch.rand(num_samples, 1) * 100,      # 0-100
            "cognitive_load": torch.rand(num_samples, 1) * 100,
            "neuro_risk": torch.rand(num_samples, 1) * 100,
            "bio_score": torch.rand(num_samples, 1) * 100
        }
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        t = {k: v[idx] for k, v in self.targets.items()}
        return (self.audio_hc[idx], self.audio_deep[idx], self.rppg_sig[idx], 
                self.rppg_hrv[idx], self.gaze_seq[idx], self.keystroke_seq[idx], t)

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 8 — Training Pipeline (Gaussian NLL Loss)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gaussian_nll_loss(pred, target, variance):
    """
    Gaussian NLL training the model to say "I predict X, but I am uncertain (variance)".
    NLL = 0.5 * log(variance) + 0.5 * (target - pred)^2 / variance
    """
    loss = 0.5 * torch.log(variance) + 0.5 * ((target - pred) ** 2) / variance
    return loss.mean()

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    weights = {"neuro_risk": 2.0, "bio_score": 2.0} # Custom weights from implementation plan
    
    for batch_idx, data in enumerate(loader):
        (ahc, adp, r_sig, r_hrv, gaze, keys, tgts) = data
        ahc, adp = ahc.to(device), adp.to(device)
        r_sig, r_hrv = r_sig.to(device), r_hrv.to(device)
        gaze, keys = gaze.to(device), keys.to(device)
        tgts = {k: v.to(device) for k, v in tgts.items()}
        
        optimizer.zero_grad()
        # Mixed Precision logic could be added here (torch.cuda.amp)
        _, preds, uncs = model(ahc, adp, r_sig, r_hrv, gaze, keys)
        
        batch_loss = 0.0
        for name in tgts.keys():
            w = weights.get(name, 1.0)
            loss_t = gaussian_nll_loss(preds[name], tgts[name], uncs[name])
            batch_loss += loss_t * w
            
        batch_loss.backward()
        # Grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        
    return total_loss / len(loader)

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 9 — Evaluation & Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0
    mae_metrics = {name: 0.0 for name in model.target_names}
    
    with torch.no_grad():
        for data in loader:
            (ahc, adp, r_sig, r_hrv, gaze, keys, tgts) = data
            ahc, adp = ahc.to(device), adp.to(device)
            r_sig, r_hrv = r_sig.to(device), r_hrv.to(device)
            gaze, keys = gaze.to(device), keys.to(device)
            tgts = {k: v.to(device) for k, v in tgts.items()}
            
            _, preds, uncs = model(ahc, adp, r_sig, r_hrv, gaze, keys)
            
            for name in tgts.keys():
                loss_t = gaussian_nll_loss(preds[name], tgts[name], uncs[name])
                val_loss += loss_t.item()
                mae_metrics[name] += torch.abs(preds[name] - tgts[name]).mean().item()
                
    n_batches = len(loader)
    mae_metrics = {k: v / n_batches for k,v in mae_metrics.items()}
    return val_loss / n_batches, mae_metrics

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 10 — Bio Signature Tracker (Mahalanobis Drift)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BioSignatureTracker:
    def __init__(self):
        self.baseline_signatures = []
        self.inv_cov_matrix = None
        self.mean_vector = None
        
    def add_baseline(self, signature_vec):
        """Add weekly baseline signatures (e.g. over 4 weeks)"""
        self.baseline_signatures.append(signature_vec.detach().cpu().numpy())
        
    def calibrate(self):
        """Compute mean and inverse covariance matrix"""
        data = np.vstack(self.baseline_signatures)
        self.mean_vector = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-4 # Add jitter
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        
    def compute_drift(self, new_signature):
        """Calculate Mahalanobis distance from baseline"""
        vec = new_signature.detach().cpu().numpy() - self.mean_vector
        mah_dist = np.sqrt(vec @ self.inv_cov_matrix @ vec.T)
        return mah_dist[0][0]

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 11 — ONNX Export & Benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_to_onnx(model, cfg):
    model.eval()
    
    # Dummy inputs
    b = 1 # batch size 1 for inference
    ahc = torch.randn(b, cfg.audio_handcrafted_dim, device=cfg.device)
    adp = torch.randn(b, cfg.audio_deep_dim, device=cfg.device)
    r_sig = torch.randn(b, 1, 300, device=cfg.device)
    r_hrv = torch.randn(b, cfg.rppg_hr_features_dim, device=cfg.device)
    gaze = torch.randn(b, cfg.gaze_seq_len, cfg.gaze_feat_dim, device=cfg.device)
    keys = torch.randn(b, cfg.keystroke_seq_len, cfg.keystroke_feat_dim, device=cfg.device)
    
    dummy_input = (ahc, adp, r_sig, r_hrv, gaze, keys)
    
    export_path = "bioecho_model.onnx"
    torch.onnx.export(model, dummy_input, export_path,
                      export_params=True, opset_version=14,
                      do_constant_folding=True,
                      input_names=['audio_hc', 'audio_deep', 'rppg_sig', 'rppg_hrv', 'gaze_seq', 'keystroke_seq'],
                      output_names=['bio_sig', 'preds', 'uncs'],
                      dynamic_axes={'audio_hc': {0 : 'batch_size'}, 'bio_sig': {0 : 'batch_size'}})
    print(f"[ONNX] Model exported successfully to {export_path}")

# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 12 — Execution / Main Block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("Initialize BioEcho Model Prototype...")
    device = config.device
    model = BioEchoModel(config).to(device)
    
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 1. Create Datasets
    print("Generating Synthetic Data...")
    train_dataset = BioEchoDataset(2000, config)
    val_dataset = BioEchoDataset(400, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 2. Setup Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.audio_enc.parameters(), 'lr': config.base_lr * 0.3},
        {'params': model.rppg_enc.parameters(), 'lr': config.base_lr * 0.3},
        {'params': model.gaze_enc.parameters(), 'lr': config.base_lr * 0.3},
        {'params': model.keystroke_enc.parameters(), 'lr': config.base_lr * 0.3},
        {'params': model.fusion.parameters(), 'lr': config.base_lr * 1.0},
        {'params': model.pred_heads.parameters(), 'lr': config.base_lr * 2.0},
        {'params': model.unc_heads.parameters(), 'lr': config.base_lr * 2.0}
    ], weight_decay=1e-4)
    
    # 3. Quick Run (1 Epoch for prototyping)
    print("Training 1 Epoch...")
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_mae = evaluate(model, val_loader, device)
    print(f"Epoch 1 Time: {time.time()-start_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # 4. ONNX Export
    print("Testing ONNX export for AMD Slingshot...")
    try:
        export_to_onnx(model, config)
    except Exception as e:
        print(f"ONNX export exception: {e}")
    
    # 5. Mahalanobis Test
    print("Testing Mahalanobis Drift tracker...")
    tracker = BioSignatureTracker()
    for i in range(10): # simulate 10 weeks of baseline
        hc, dp, sig, hrv, g, k, _ = train_dataset[i]
        with torch.no_grad():
            bio_sig, _, _ = model(hc.unsqueeze(0).to(device), dp.unsqueeze(0).to(device), 
                                  sig.unsqueeze(0).to(device), hrv.unsqueeze(0).to(device), 
                                  g.unsqueeze(0).to(device), k.unsqueeze(0).to(device))
            tracker.add_baseline(bio_sig)
            
    tracker.calibrate()
    
    hc, dp, sig, hrv, g, k, _ = val_dataset[0] # completely new vector
    with torch.no_grad():
        new_bio_sig, _, _ = model(hc.unsqueeze(0).to(device), dp.unsqueeze(0).to(device), 
                                  sig.unsqueeze(0).to(device), hrv.unsqueeze(0).to(device), 
                                  g.unsqueeze(0).to(device), k.unsqueeze(0).to(device))
        drift = tracker.compute_drift(new_bio_sig)
    print(f"Simulated Drift Distance (Mahalanobis): {drift:.4f}")
    print("✅ Phase 1 Prototype Completed successfully.")
    print("Ready for Kaggle deployment and Deep Model Weights.")
