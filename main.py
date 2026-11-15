#!/usr/bin/env python3
"""
main.py - Urban Sound Event Detection with Deep Residual SE-CNN & Mel Spectrograms

Project 91: PyTorch 100 Projects Challenge
Detects urban sound events across 10 classes:
[air_conditioner, car_horn, children_playing, dog_bark, drilling,
 engine_idling, gun_shot, jackhammer, siren, street_music]
"""

import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio.transforms as T
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

URBAN_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(URBAN_CLASSES)}
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "urban_sound_cnn_best.pth")


class UrbanSoundSynthesizer:
    """
    Physical-acoustic synthesis for 10 urban sound event categories.
    Generates realistic 22050 Hz 3.0s audio waveforms.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sr = sample_rate
        self.duration = duration
        self.n = int(self.sr * self.duration)
        self.t = torch.linspace(0, self.duration, self.n)

    def synthesize(self, event_class: str, seed: int = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t = self.t
        audio = torch.zeros(self.n)

        if event_class == "air_conditioner":
            # Low hum (60 & 120 Hz) + broadband fan airflow
            drone = 0.35 * torch.sin(2 * np.pi * 60.0 * t) + 0.20 * torch.sin(2 * np.pi * 120.0 * t)
            noise = torch.randn(self.n) * 0.25
            env = 0.85 + 0.15 * torch.sin(2 * np.pi * 0.25 * t)
            audio = (drone + noise) * env

        elif event_class == "car_horn":
            # Dual-tone harmonic horn blast (F4 349Hz, A4 440Hz)
            start_idx = int(0.6 * self.sr)
            horn_len = int(1.2 * self.sr)
            ht = torch.linspace(0, 1.2, horn_len)
            h_env = torch.ones(horn_len)
            attack = int(0.04 * self.sr)
            release = int(0.08 * self.sr)
            h_env[:attack] = torch.linspace(0, 1, attack)
            h_env[-release:] = torch.linspace(1, 0, release)
            tone1 = torch.sin(2 * np.pi * 349.2 * ht) + 0.3 * torch.sin(2 * np.pi * 698.4 * ht)
            tone2 = torch.sin(2 * np.pi * 440.0 * ht) + 0.25 * torch.sin(2 * np.pi * 880.0 * ht)
            horn_sig = (tone1 + tone2) * 0.60 * h_env
            audio[start_idx : start_idx + horn_len] = horn_sig
            audio = audio + torch.randn(self.n) * 0.05

        elif event_class == "children_playing":
            # High-pitched dynamic vocal shouts and laughter
            for p in [0.4, 1.1, 1.9, 2.5]:
                idx = int(p * self.sr)
                burst_len = int(0.35 * self.sr)
                bt = torch.linspace(0, 0.35, burst_len)
                f0 = 550.0 + 200.0 * torch.sin(2 * np.pi * 6.0 * bt)
                vocal = (torch.sin(2 * np.pi * f0 * bt) + 0.4 * torch.sin(4 * np.pi * f0 * bt)) * torch.exp(-bt * 5.0)
                audio[idx : idx + burst_len] += vocal * 0.65
            audio = audio + torch.randn(self.n) * 0.08

        elif event_class == "dog_bark":
            # Rapid transient bursts with harmonic formants (400-800 Hz)
            bark_times = [0.3, 0.9, 1.6, 2.3]
            for bp in bark_times:
                idx = int(bp * self.sr)
                bk_len = int(0.22 * self.sr)
                bkt = torch.linspace(0, 0.22, bk_len)
                pitch_sweep = 480.0 * torch.exp(-bkt * 6.0)
                tone = torch.sin(2 * np.pi * pitch_sweep * bkt) + 0.5 * torch.sin(4 * np.pi * pitch_sweep * bkt)
                noise_growl = torch.randn(bk_len) * 0.35
                bark = (tone + noise_growl) * torch.exp(-bkt * 14.0) * 0.85
                audio[idx : idx + bk_len] += bark
            audio = audio + torch.randn(self.n) * 0.03

        elif event_class == "drilling":
            # Motor whine (320 Hz) with rapid mechanical impact modulation
            mod = 0.5 + 0.5 * torch.sin(2 * np.pi * 85.0 * t)
            whine = (torch.sin(2 * np.pi * 320.0 * t) + 0.4 * torch.sin(2 * np.pi * 640.0 * t)) * mod
            friction = (torch.randn(self.n) * 0.30) * mod
            audio = (whine * 0.55 + friction * 0.45)

        elif event_class == "engine_idling":
            # Low-frequency rhythmic combustive pulses (30-40 Hz)
            f_idle = 32.0 + 3.0 * torch.sin(2 * np.pi * 0.5 * t)
            pulses = (torch.sin(2 * np.pi * f_idle * t) + 
                      0.6 * torch.sin(4 * np.pi * f_idle * t) + 
                      0.3 * torch.sin(6 * np.pi * f_idle * t)) * 0.70
            rumble = torch.randn(self.n) * 0.15
            audio = pulses + rumble

        elif event_class == "gun_shot":
            # High-intensity shockwave (<3ms attack) + low boom + acoustic reverberation tail
            shot_start = int(0.5 * self.sr)
            shot_len = int(1.8 * self.sr)
            st = torch.linspace(0, 1.8, shot_len)
            impulse = torch.randn(shot_len) * torch.exp(-st * 28.0) * 0.95
            boom = torch.sin(2 * np.pi * 95.0 * st) * torch.exp(-st * 8.0) * 0.80
            audio[shot_start : shot_start + shot_len] = impulse + boom
            audio = audio + torch.randn(self.n) * 0.025

        elif event_class == "jackhammer":
            # Rhythmic percussive strikes at 19 Hz
            strike_rate = 19.0
            num_strikes = int(self.duration * strike_rate)
            for s in range(num_strikes):
                idx = int((s / strike_rate) * self.sr)
                if idx + int(0.04 * self.sr) < self.n:
                    slen = int(0.04 * self.sr)
                    s_t = torch.linspace(0, 0.04, slen)
                    impact = (torch.sin(2 * np.pi * 1400.0 * s_t) + torch.randn(slen) * 0.4) * torch.exp(-s_t * 60.0)
                    thud = torch.sin(2 * np.pi * 110.0 * s_t) * torch.exp(-s_t * 40.0) * 0.7
                    audio[idx : idx + slen] += (impact + thud) * 0.75

        elif event_class == "siren":
            # Two-tone or sweeping wail between 650 Hz and 1150 Hz
            f_siren = 900.0 + 250.0 * torch.sin(2 * np.pi * 1.2 * t)
            phase = 2 * np.pi * torch.cumsum(f_siren, dim=0) / self.sr
            siren_tone = torch.sin(phase) * 0.70 + 0.25 * torch.sin(2 * phase)
            audio = siren_tone + torch.randn(self.n) * 0.05

        elif event_class == "street_music":
            # Harmonic chords (C major: C4 261.6Hz, E4 329.6Hz, G4 392.0Hz) + rhythm
            chord = (torch.sin(2 * np.pi * 261.6 * t) * 0.35 +
                     torch.sin(2 * np.pi * 329.6 * t) * 0.30 +
                     torch.sin(2 * np.pi * 392.0 * t) * 0.25)
            rhythm = 0.7 + 0.3 * torch.sin(2 * np.pi * 2.0 * t)
            audio = chord * rhythm + torch.randn(self.n) * 0.04

        # Peak normalization
        peak = torch.max(torch.abs(audio))
        if peak > 0.88:
            audio = audio / peak * 0.88

        return audio.clamp(-1.0, 1.0)


class LogMelSpectrogramExtractor:
    """Computes normalized Log-Mel spectrograms from audio waveforms."""
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.mel_tf = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate // 2
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        # Compute Log-Mel
        mel = torch.log(self.mel_tf(audio) + 1e-6)
        # Instance normalization
        mean = mel.mean(dim=[-2, -1], keepdim=True)
        std = mel.std(dim=[-2, -1], keepdim=True) + 1e-6
        return (mel - mean) / std


class SpecAugment(nn.Module):
    """Applies time and frequency masking to spectrograms."""
    def __init__(self, max_freq_mask=16, max_time_mask=24):
        super().__init__()
        self.max_f = max_freq_mask
        self.max_t = max_time_mask

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        out = spec.clone()
        b, c, f, t = out.shape
        # Frequency masking
        f_len = np.random.randint(0, self.max_f)
        f_pos = np.random.randint(0, max(1, f - f_len))
        out[:, :, f_pos : f_pos + f_len, :] = 0.0
        # Time masking
        t_len = np.random.randint(0, self.max_t)
        t_pos = np.random.randint(0, max(1, t - t_len))
        out[:, :, :, t_pos : t_pos + t_len] = 0.0
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SqueezeExcitation(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class UrbanSoundResNet(nn.Module):
    """
    Deep Residual SE-CNN with Dual Attentive Pooling for Urban Sound Event Detection.
    """
    def __init__(self, num_classes: int = len(URBAN_CLASSES)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResidualSEBlock(32, 64, stride=(2, 2))   # 128x259 -> 64x130
        self.stage2 = ResidualSEBlock(64, 128, stride=(2, 2))  # 64x130 -> 32x65
        self.stage3 = ResidualSEBlock(128, 256, stride=(2, 2)) # 32x65 -> 16x33

        # Dual Pooling across frequency & time (Avg + Max for transient capture)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1) # [B, 1, F, T]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        avg_f = self.avg_pool(x).flatten(1)
        max_f = self.max_pool(x).flatten(1)
        feat = torch.cat([avg_f, max_f], dim=1)

        logits = self.classifier(feat)
        return logits


class UrbanSoundDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_dataset(synthesizer: UrbanSoundSynthesizer, extractor: LogMelSpectrogramExtractor, n_train=30, n_val=10):
    """Generates synthetic multi-class urban sound dataset."""
    train_samples = []
    val_samples = []
    canonical_audios = {}

    for c_idx, c_name in enumerate(URBAN_CLASSES):
        # 1. Canonical reference clip
        ref_audio = synthesizer.synthesize(c_name, seed=42 + c_idx)
        canonical_audios[c_name] = ref_audio
        ref_path = os.path.join(OUTPUTS_DIR, f"{c_name}_event.wav")
        sf.write(ref_path, ref_audio.numpy(), SAMPLE_RATE)

        # 2. Train samples
        for i in range(n_train):
            audio = synthesizer.synthesize(c_name, seed=1000 + c_idx * 100 + i)
            # Add light random noise
            audio = audio + torch.randn(len(audio)) * np.random.uniform(0.01, 0.05)
            spec = extractor(audio).squeeze(0) # [128, T]
            train_samples.append((spec, c_idx))

        # 3. Val samples
        for i in range(n_val):
            audio = synthesizer.synthesize(c_name, seed=5000 + c_idx * 100 + i)
            audio = audio + torch.randn(len(audio)) * np.random.uniform(0.01, 0.05)
            spec = extractor(audio).squeeze(0)
            val_samples.append((spec, c_idx))

    return train_samples, val_samples, canonical_audios


def train_model(model, train_loader, val_loader, epochs=12):
    print(f"\n[Training] Training UrbanSoundResNet for {epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    augment = SpecAugment()

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for specs, labels in train_loader:
            specs = augment(specs.unsqueeze(1))
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        ep_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(ep_loss)
        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                logits = model(specs.unsqueeze(1))
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == labels).sum().item())

        val_acc = (correct / len(val_loader.dataset)) * 100.0
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)

        if ep % 2 == 0 or ep == epochs:
            print(f"  Epoch {ep:02d}/{epochs:02d} | Loss: {ep_loss:.4f} | Val Accuracy: {val_acc:.1f}% (Best: {best_acc:.1f}%)")

    print(f"[Training Complete] Checkpoint saved: {MODEL_CHECKPOINT_PATH} (Best Acc: {best_acc:.1f}%)")
    return history


def evaluate_and_benchmark(model, val_loader, extractor, canonical_audios):
    print("\n[Evaluation] Calculating classification metrics & running latency benchmark...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for specs, labels in val_loader:
            logits = model(specs.unsqueeze(1))
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=range(len(URBAN_CLASSES)))
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    per_class = {}
    p_c, r_c, f_c, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(URBAN_CLASSES)), zero_division=0)
    for idx, c_name in enumerate(URBAN_CLASSES):
        per_class[c_name] = {
            "precision": round(float(p_c[idx]), 4),
            "recall": round(float(r_c[idx]), 4),
            "f1_score": round(float(f_c[idx]), 4)
        }

    # Latency Benchmark (50 trials)
    test_wav = canonical_audios["gun_shot"]
    test_spec = extractor(test_wav).unsqueeze(0)
    for _ in range(5):
        with torch.no_grad():
            _ = model(test_spec)

    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        with torch.no_grad():
            _ = model(test_spec)
    t1 = time.perf_counter()

    avg_latency_s = (t1 - t0) / trials
    avg_latency_ms = avg_latency_s * 1000.0
    rtf = avg_latency_s / DURATION
    speedup = 1.0 / rtf if rtf > 0 else 0
    headroom = max(0.0, (1.0 - rtf) * 100.0)

    metrics = {
        "project": "Project 91: Urban Sound Event Detection",
        "taxonomy": URBAN_CLASSES,
        "classification": {
            "accuracy_pct": round(acc, 2),
            "macro_precision_pct": round(prec * 100.0, 2),
            "macro_recall_pct": round(rec * 100.0, 2),
            "macro_f1_pct": round(f1 * 100.0, 2),
            "per_class": per_class
        },
        "latency_benchmark": {
            "mean_latency_ms": round(avg_latency_ms, 2),
            "real_time_factor_rtf": round(rtf, 4),
            "speedup_factor": round(speedup, 1),
            "cpu_headroom_pct": round(headroom, 1)
        }
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "urban_sound_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, cm_norm


def generate_diagnostic_report(canonical_audios, cm_norm, metrics, extractor, model):
    """Renders 6-panel visual diagnostic report."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

    # Panel 1: Waveforms of Key Urban Sound Events
    ax1 = fig.add_subplot(gs[0, 0])
    t_axis = np.linspace(0, DURATION, NUM_SAMPLES)
    sample_keys = ["gun_shot", "siren", "dog_bark", "car_horn", "jackhammer"]
    colors = ["#E53E3E", "#3182CE", "#DD6B20", "#38A169", "#805AD5"]
    for i, k in enumerate(sample_keys):
        wav = canonical_audios[k].numpy()
        offset = (len(sample_keys) - 1 - i) * 1.6
        ax1.plot(t_axis, wav * 0.7 + offset, color=colors[i], alpha=0.85)
        ax1.text(0.05, offset + 0.55, k.replace("_", " ").title(), fontsize=9, fontweight="bold", color="#2D3748")
    ax1.set_yticks([])
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Panel 1: Urban Sound Event Waveforms", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: Mel Spectrograms (Gunshot vs Siren)
    ax2 = fig.add_subplot(gs[0, 1])
    spec_gun = extractor(canonical_audios["gun_shot"]).squeeze(0).numpy()
    spec_siren = extractor(canonical_audios["siren"]).squeeze(0).numpy()
    stacked = np.vstack([spec_gun, spec_siren])
    im2 = ax2.imshow(stacked, origin="lower", aspect="auto", cmap="magma")
    ax2.axhline(128, color="white", linestyle="--", linewidth=1.5)
    ax2.text(5, 135, "Gunshot (Impulsive Shockwave & Decay)", color="white", fontsize=8, fontweight="bold")
    ax2.text(5, 10, "Siren (Sinusoidal Frequency Modulation)", color="white", fontsize=8, fontweight="bold")
    ax2.set_title("Panel 2: Mel-Spectrogram Fingerprints (Gunshot vs Siren)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Mel Frequency Bins")
    fig.colorbar(im2, ax=ax2, pad=0.02)

    # Panel 3: 10x10 Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = [c.replace("_", "\n").title() for c in URBAN_CLASSES]
    ax3.set_xticks(range(10))
    ax3.set_yticks(range(10))
    ax3.set_xticklabels(ticks, fontsize=7, rotation=45, ha="right")
    ax3.set_yticklabels(ticks, fontsize=7)
    ax3.set_title("Panel 3: Normalized Confusion Matrix (10 Classes)", fontsize=11, fontweight="bold")
    fig.colorbar(im3, ax=ax3, pad=0.02)

    # Panel 4: Per-Class F1-Scores
    ax4 = fig.add_subplot(gs[1, 0])
    per_class = metrics["classification"]["per_class"]
    f1_vals = [per_class[c]["f1_score"] * 100.0 for c in URBAN_CLASSES]
    names = [c.replace("_", " ").title() for c in URBAN_CLASSES]
    bars = ax4.barh(names, f1_vals, color="#3182CE")
    ax4.set_xlim(0, 115)
    ax4.set_xlabel("F1-Score (%)")
    ax4.set_title("Panel 4: Per-Class F1 Detection Performance", fontsize=11, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.3, axis="x")

    # Panel 5: Event Detection Probability Distribution (Gunshot test)
    ax5 = fig.add_subplot(gs[1, 1])
    test_spec = extractor(canonical_audios["gun_shot"]).unsqueeze(0)
    with torch.no_grad():
        logits = model(test_spec)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy() * 100.0
    y_pos = np.arange(len(URBAN_CLASSES))
    colors_p = ["#E53E3E" if c == "gun_shot" else "#4A5568" for c in URBAN_CLASSES]
    ax5.bar(y_pos, probs, color=colors_p)
    ax5.set_xticks(y_pos)
    ax5.set_xticklabels([c.replace("_", "\n").title() for c in URBAN_CLASSES], fontsize=7)
    ax5.set_ylabel("Posterior Probability (%)")
    ax5.set_title("Panel 5: Prediction Confidence Distribution (Gunshot)", fontsize=11, fontweight="bold")
    ax5.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Panel 6: Radar Profile
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    categories = ["Accuracy\n(%)", "Macro F1\n(%)", "Speedup\n(x10)", "Headroom\n(%)", "Coverage\n(10 Cls)"]
    acc_v = metrics["classification"]["accuracy_pct"]
    f1_v = metrics["classification"]["macro_f1_pct"]
    spd_v = min(100.0, metrics["latency_benchmark"]["speedup_factor"] * 0.5)
    hdr_v = metrics["latency_benchmark"]["cpu_headroom_pct"]
    cov_v = 100.0

    vals = [acc_v, f1_v, spd_v, hdr_v, cov_v]
    vals += vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax6.plot(angles, vals, color="#3182CE", linewidth=2)
    ax6.fill(angles, vals, color="#3182CE", alpha=0.25)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=8, fontweight="bold")
    ax6.set_ylim(0, 105)
    ax6.set_title("Panel 6: System Capability Radar", fontsize=11, fontweight="bold", pad=15)

    report_path = os.path.join(OUTPUTS_DIR, "urban_sound_report.png")
    plt.savefig(report_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Report] Diagnostic report saved to: {report_path}")


def main():
    print("=" * 70)
    print(" PROJECT 91: URBAN SOUND EVENT DETECTION (URBANSOUND8K TAXONOMY)")
    print("=" * 70)

    synthesizer = UrbanSoundSynthesizer()
    extractor = LogMelSpectrogramExtractor()

    print("[Stage 1] Synthesizing 10-Class Urban Sound Dataset & Canonical Audios...")
    train_data, val_data, canonical_audios = build_dataset(synthesizer, extractor, n_train=30, n_val=10)
    print(f"  Dataset: {len(train_data)} Train samples, {len(val_data)} Validation samples")

    train_loader = DataLoader(UrbanSoundDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(UrbanSoundDataset(val_data), batch_size=32, shuffle=False)

    print("[Stage 2] Building Deep Residual SE-CNN Architecture...")
    model = UrbanSoundResNet(num_classes=len(URBAN_CLASSES))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {param_count:,}")

    history = train_model(model, train_loader, val_loader, epochs=8)

    # Load best checkpoint
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))

    print("[Stage 3] Evaluating Metrics & Generating Benchmark Dashboard...")
    metrics, cm_norm = evaluate_and_benchmark(model, val_loader, extractor, canonical_audios)
    generate_diagnostic_report(canonical_audios, cm_norm, metrics, extractor, model)

    print("\n" + "=" * 70)
    print(" URBAN SOUND EVENT DETECTION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Accuracy           : {metrics['classification']['accuracy_pct']}%")
    print(f"  Macro F1-Score     : {metrics['classification']['macro_f1_pct']}%")
    print(f"  Mean Latency       : {metrics['latency_benchmark']['mean_latency_ms']} ms")
    print(f"  Real-Time Factor   : {metrics['latency_benchmark']['real_time_factor_rtf']}x")
    print(f"  Processing Speedup : {metrics['latency_benchmark']['speedup_factor']}x")
    print(f"  CPU Headroom       : {metrics['latency_benchmark']['cpu_headroom_pct']}%")
    print("=" * 70)
    print(" PROJECT 91 PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
