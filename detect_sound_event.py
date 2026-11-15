#!/usr/bin/env python3
"""
detect_sound_event.py - Standalone CLI for Urban Sound Event Detection & Safety Alerting

Usage examples:
  python detect_sound_event.py --audio outputs/gun_shot_event.wav
  python detect_sound_event.py --audio outputs/siren_event.wav --top-k 3 --benchmark
  python detect_sound_event.py --audio outputs/dog_bark_event.wav --output-json outputs/pred.json
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    UrbanSoundResNet,
    LogMelSpectrogramExtractor,
    URBAN_CLASSES,
    SAMPLE_RATE,
    DURATION,
    NUM_SAMPLES,
    MODELS_DIR,
    OUTPUTS_DIR
)

DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "urban_sound_cnn_best.pth")
SAFETY_CRITICAL_CLASSES = ["gun_shot", "siren", "car_horn"]


def load_and_preprocess(audio_path: str, target_sr: int = SAMPLE_RATE, target_duration: float = DURATION):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    waveform = torch.from_numpy(data)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    req_len = int(target_sr * target_duration)
    if waveform.shape[0] < req_len:
        waveform = F.pad(waveform, (0, req_len - waveform.shape[0]))
    elif waveform.shape[0] > req_len:
        waveform = waveform[:req_len]

    peak = torch.max(torch.abs(waveform))
    if peak > 1e-4:
        waveform = waveform / peak * 0.90

    return waveform


def detect_events(
    audio_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    top_k: int = 3,
    alert_threshold: float = 0.50,
    benchmark: bool = False,
    output_json: str = None
):
    print("=" * 70)
    print(" URBAN SOUND EVENT DETECTOR - DEEP RESIDUAL SE-CNN INFERENCE")
    print("=" * 70)
    print(f"Target Audio   : {audio_path}")
    print(f"Model Checkpoint: {checkpoint_path}")

    # 1. Load Audio
    waveform = load_and_preprocess(audio_path)
    audio_dur = len(waveform) / SAMPLE_RATE
    print(f"Loaded Clip    : {audio_dur:.2f}s ({len(waveform)} samples @ {SAMPLE_RATE} Hz)")

    # 2. Extract Log-Mel Spectrogram
    extractor = LogMelSpectrogramExtractor()
    with torch.no_grad():
        spec = extractor(waveform).unsqueeze(0) # [1, 1, 128, T]
    print(f"Spectrogram Dim: {list(spec.shape)}")

    # 3. Load Model
    model = UrbanSoundResNet(num_classes=len(URBAN_CLASSES))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Run main.py first.")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # 4. Inference
    with torch.no_grad():
        logits = model(spec)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    topk_probs, topk_indices = torch.topk(probs, k=min(top_k, len(URBAN_CLASSES)))
    top_class = URBAN_CLASSES[topk_indices[0].item()]
    top_conf = topk_probs[0].item() * 100.0

    print("\n" + "-" * 70)
    print(f" DETECTED URBAN SOUND EVENT: [{top_class.replace('_', ' ').upper()}] ({top_conf:.2f}% Confidence)")
    print("-" * 70)
    print(f"{'Rank':<6}{'Sound Event Class':<24}{'Confidence':<16}{'Visual Probability Bar'}")
    print("-" * 70)

    topk_results = []
    for rank, (idx, p) in enumerate(zip(topk_indices, topk_probs), start=1):
        name = URBAN_CLASSES[idx.item()]
        pct = p.item() * 100.0
        bar_len = int(pct / 3.0)
        bar = "█" * bar_len + "░" * (33 - bar_len)
        print(f" #{rank:<5}{name.replace('_', ' ').title():<24}{pct:>6.2f}%         {bar}")
        topk_results.append({
            "rank": rank,
            "class": name,
            "confidence_pct": round(pct, 2)
        })

    # Alert Check
    alert_triggered = False
    if top_class in SAFETY_CRITICAL_CLASSES and (top_conf / 100.0) >= alert_threshold:
        alert_triggered = True
        print("\n" + "!" * 70)
        print(f" [SAFETY ALERT TRIGGERED]: High-priority critical event detected: {top_class.upper()}!")
        print("!" * 70)

    # 5. Optional Benchmark
    bench_data = None
    if benchmark:
        print("\n[Benchmarking] Executing 50-trial latency & RTF benchmark...")
        trials = 50
        for _ in range(5):
            with torch.no_grad():
                _ = model(spec)

        t0 = time.perf_counter()
        for _ in range(trials):
            with torch.no_grad():
                _ = model(spec)
        t1 = time.perf_counter()

        avg_lat_s = (t1 - t0) / trials
        avg_lat_ms = avg_lat_s * 1000.0
        rtf = avg_lat_s / audio_dur
        speedup = 1.0 / rtf if rtf > 0 else 0
        headroom = max(0.0, (1.0 - rtf) * 100.0)

        print(f"  -> Mean Inference Latency : {avg_lat_ms:.2f} ms")
        print(f"  -> Real-Time Factor (RTF) : {rtf:.4f}x")
        print(f"  -> Processing Speedup     : {speedup:.1f}x Faster than Real Time")
        print(f"  -> CPU Processing Headroom: {headroom:.1f}%")

        bench_data = {
            "mean_latency_ms": round(avg_lat_ms, 2),
            "real_time_factor": round(rtf, 4),
            "speedup_factor": round(speedup, 1),
            "cpu_headroom_pct": round(headroom, 1)
        }

    payload = {
        "audio_path": audio_path,
        "detected_event": top_class,
        "confidence_pct": round(top_conf, 2),
        "safety_alert_triggered": alert_triggered,
        "top_k_results": topk_results,
        "benchmark": bench_data
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Saved] Results exported to: {output_json}")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Urban Sound Event Detection CLI.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to trained PyTorch model")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--alert-threshold", type=float, default=0.50, help="Confidence threshold for safety alerts")
    parser.add_argument("--benchmark", action="store_true", help="Run 50-trial latency & RTF benchmark")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to output JSON file")

    args = parser.parse_args()
    detect_events(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        alert_threshold=args.alert_threshold,
        benchmark=args.benchmark,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()
