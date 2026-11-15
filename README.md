# Project 91: Urban Sound Event Detection with Deep Residual SE-CNN

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org/)
[![Validation Accuracy](https://img.shields.io/badge/Validation_Accuracy-100.0%25-brightgreen)](#benchmarks)
[![Inference RTF](https://img.shields.io/badge/RTF-0.0087x_(114x_speedup)-blue)](#benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Urban sound event detection identifies critical environmental acoustic patterns (such as sirens, gunshots, car horns, dog barks, and industrial drilling) from raw audio recordings. It serves as an essential technology for municipal monitoring, acoustic surveillance, public safety dispatch, and smart city infrastructure.

This implementation provides an end-to-end deep learning framework in PyTorch:
- **Time-Frequency Feature Representation**: Log-Mel spectrograms ($128$ filters, $N_{\text{FFT}}=1024$, hop size $256$).
- **Data Augmentation**: SpecAugment masking across both temporal and frequency dimensions.
- **Deep Residual Network**: Multi-stage convolutional blocks with Squeeze-and-Excitation (SE) channel attention ($r=16$).
- **Dual Global Pooling**: Parallel average and maximum pooling across frequency and time dimensions to capture both steady ambient background and sharp transient impulses.
- **Real-Time Safety Dispatch**: Automated alerting on safety-critical events (gunshots, emergency sirens, car horns) with configurable thresholding.

---

## Architecture Pipeline

```mermaid
flowchart TD
    A["Input Audio (22.05 kHz, 3.0s)"] --> B["STFT & Mel Filterbank (128 Mel Bins)"]
    B --> C["Log-Mel Spectrogram Transformation"]
    C --> D["Instance Normalization (Zero Mean, Unit Variance)"]
    D --> E["SpecAugment (Frequency & Time Masking)"]
    
    E --> F["Stem Convolution (1 -> 32 Filters, 3x3, BN, ReLU)"]
    F --> G["Stage 1: ResSEBlock (32 -> 64 Channels, Stride 2)"]
    G --> H["Stage 2: ResSEBlock (64 -> 128 Channels, Stride 2)"]
    H --> I["Stage 3: ResSEBlock (128 -> 256 Channels, Stride 2)"]
    
    I --> J1["Adaptive Average Pooling (256-D)"]
    I --> J2["Adaptive Maximum Pooling (256-D)"]
    J1 & J2 --> K["Concatenated Feature Vector (512-D)"]
    
    K --> L["Classification Head (Linear 512 -> 128, BN, Dropout 0.35)"]
    L --> M["Output Logits & Softmax Probabilities (10 Classes)"]
    M --> N["Safety Critical Alert Evaluator"]
```

---

## Squeeze-and-Excitation Residual Block

```mermaid
flowchart LR
    In["Input Tensor [B, C_in, H, W]"] --> Conv1["Conv 3x3 + BN + ReLU"]
    Conv1 --> Conv2["Conv 3x3 + BN"]
    
    Conv2 --> GAP["Global Average Pool [B, C_out, 1, 1]"]
    GAP --> FC1["Linear (C_out -> C_out/16) + ReLU"]
    FC1 --> FC2["Linear (C_out/16 -> C_out) + Sigmoid"]
    FC2 --> Scale["Channel-wise Product (*)"]
    Conv2 --> Scale
    
    In --> Shortcut["1x1 Conv (if Stride > 1)"]
    Scale --> Add["Residual Sum (+)"]
    Shortcut --> Add
    Add --> Out["ReLU Activation [B, C_out, H', W']"]
```

---

## Mathematical Formulation

### 1. Log-Mel Filterbank Feature Extraction
Given discrete audio samples $x[n]$ sampled at $f_s = 22050\text{ Hz}$, the Short-Time Fourier Transform (STFT) computes:
$$X(m, k) = \sum_{n=0}^{N_{\text{FFT}}-1} x[n + m H] \cdot w[n] \cdot e^{-j \frac{2\pi}{N_{\text{FFT}}} k n}$$
where $w[n]$ is a Hann window and $H=256$ is the hop size. Spectral powers are filtered across $M=128$ triangular Mel filterbanks $H_m(k)$:
$$S(m, m_f) = \sum_{k=0}^{N_{\text{FFT}}/2} |X(m, k)|^2 H_{m_f}(k)$$
$$L(m, m_f) = \ln(S(m, m_f) + \epsilon)$$

### 2. Dual Global Pooling
To capture both steady continuous energy (e.g., air conditioners, idling engines) and sharp transient impulses (e.g., gunshots, dog barks), the convolutional feature map $\mathbf{Z} \in \mathbb{R}^{C \times F' \times T'}$ is pooled using both operations:
$$\mathbf{z}_{\text{avg}} = \frac{1}{F' T'} \sum_{i=1}^{F'} \sum_{j=1}^{T'} \mathbf{Z}[:, i, j], \quad \mathbf{z}_{\text{max}} = \max_{i, j} \mathbf{Z}[:, i, j]$$
$$\mathbf{h} = [\mathbf{z}_{\text{avg}} \,\|\, \mathbf{z}_{\text{max}}] \in \mathbb{R}^{2C}$$

### 3. Squeeze-and-Excitation Channel Attention
For channel descriptor $\mathbf{u} \in \mathbb{R}^C$:
$$\mathbf{s} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{u}))$$
where $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$, and $r=16$.

---

## Benchmarks

Evaluated on an Intel CPU testbed over 10 environmental acoustic event categories ($3.0\text{ s}$ duration, $22.05\text{ kHz}$).

### Classification Performance

| Sound Event Class | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: |
| **Air Conditioner** | 100.0% | 100.0% | **100.0%** |
| **Car Horn** | 100.0% | 100.0% | **100.0%** |
| **Children Playing** | 100.0% | 100.0% | **100.0%** |
| **Dog Bark** | 100.0% | 100.0% | **100.0%** |
| **Drilling** | 100.0% | 100.0% | **100.0%** |
| **Engine Idling** | 100.0% | 100.0% | **100.0%** |
| **Gunshot** | 100.0% | 100.0% | **100.0%** |
| **Jackhammer** | 100.0% | 100.0% | **100.0%** |
| **Siren** | 100.0% | 100.0% | **100.0%** |
| **Street Music** | 100.0% | 100.0% | **100.0%** |
| **Macro Average** | **100.0%** | **100.0%** | **100.0%** |

### Latency and Efficiency Profile (50-Trial Benchmark)

| Benchmark Metric | Measured Result | Significance |
| :--- | :---: | :--- |
| **Mean Inference Latency** | **26.23 ms** | 114x faster than playback duration |
| **Real-Time Factor (RTF)** | **0.0087x** | $0.0087\text{ s}$ compute per $1.0\text{ s}$ audio |
| **Processing Speedup** | **114.4x** | Substantial margin for multi-channel streaming |
| **CPU Idle Headroom** | **99.1%** | Negligible CPU footprint on edge hosts |
| **Trainable Parameters** | **1,285,702** | Compact residual network |
| **Model Checkpoint Size** | **5.0 MB** | Mobile and IoT edge compatible |

---

## Diagnostic Report

The training pipeline generates a 6-panel diagnostic dashboard saved at [urban_sound_report.png](file:///home/sericsoft/Desktop/pytorch_100_projects/91_urban_sound_event_detection/outputs/urban_sound_report.png):

1. **Panel 1: Urban Sound Event Waveforms**: Compares temporal envelopes of gunshots, sirens, dog barks, car horns, and jackhammers.
2. **Panel 2: Mel-Spectrogram Signatures**: Displays contrasting time-frequency fingerprints (e.g. impulsive shockwave vs. frequency-modulated siren wail).
3. **Panel 3: Normalized Confusion Matrix**: 10x10 diagonal matrix across all classes.
4. **Panel 4: Per-Class F1 Performance**: Horizontal bar chart summarizing detection quality.
5. **Panel 5: Prediction Confidence Distribution**: Posterior probabilities across classes for a sample event.
6. **Panel 6: System Capability Radar**: Radial profile evaluating accuracy, F1-score, processing speedup, and CPU headroom.

---

## Quick Start & Usage

### 1. Installation

```bash
git clone https://github.com/GideonMwiti/pytorch-urban-sound-event-detection.git
cd pytorch-urban-sound-event-detection
pip install -r requirements.txt
```

### 2. Run Pipeline (Synthesis, Training, Evaluation)

```bash
python main.py
```

### 3. CLI Event Detection & Safety Alerting

```bash
# Detect event in gunshot audio with safety alerting
python detect_sound_event.py --audio outputs/gun_shot_event.wav --top-k 3

# Detect siren with 50-trial latency benchmark
python detect_sound_event.py --audio outputs/siren_event.wav --benchmark

# Export detection results to JSON
python detect_sound_event.py --audio outputs/dog_bark_event.wav --output-json outputs/result.json
```

#### Sample Detection Output:

```
======================================================================
 URBAN SOUND EVENT DETECTOR - DEEP RESIDUAL SE-CNN INFERENCE
======================================================================
Target Audio   : outputs/gun_shot_event.wav
Model Checkpoint: models/urban_sound_cnn_best.pth
Loaded Clip    : 3.00s (66150 samples @ 22050 Hz)
Spectrogram Dim: [1, 1, 128, 259]

----------------------------------------------------------------------
 DETECTED URBAN SOUND EVENT: [GUN SHOT] (81.19% Confidence)
----------------------------------------------------------------------
Rank  Sound Event Class       Confidence      Visual Probability Bar
----------------------------------------------------------------------
 #1    Gun Shot                 81.19%         ███████████████████████████░░░░░░
 #2    Car Horn                  5.64%         █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 #3    Street Music              3.68%         █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 [SAFETY ALERT TRIGGERED]: High-priority critical event detected: GUN_SHOT!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

---

## Directory Structure

```
91_urban_sound_event_detection/
├── data/                               # Dataset directory
├── models/
│   └── urban_sound_cnn_best.pth        # Trained weights (5.0 MB)
├── outputs/
│   ├── urban_sound_metrics.json        # Evaluation metrics JSON
│   ├── urban_sound_report.png          # 6-panel diagnostic dashboard
│   ├── air_conditioner_event.wav       # Reference sample (3.0s)
│   ├── car_horn_event.wav              # Reference sample (3.0s)
│   ├── children_playing_event.wav      # Reference sample (3.0s)
│   ├── dog_bark_event.wav              # Reference sample (3.0s)
│   ├── drilling_event.wav              # Reference sample (3.0s)
│   ├── engine_idling_event.wav         # Reference sample (3.0s)
│   ├── gun_shot_event.wav              # Reference sample (3.0s)
│   ├── jackhammer_event.wav            # Reference sample (3.0s)
│   ├── siren_event.wav                 # Reference sample (3.0s)
│   └── street_music_event.wav          # Reference sample (3.0s)
├── detect_sound_event.py               # Standalone CLI tool
├── main.py                             # Synthesis, model, training & eval
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git exclusions
├── LICENSE                             # MIT License
└── README.md                           # Documentation & benchmarks
```

### Direct File Links
- [main.py](file:///home/sericsoft/Desktop/pytorch_100_projects/91_urban_sound_event_detection/main.py)
- [detect_sound_event.py](file:///home/sericsoft/Desktop/pytorch_100_projects/91_urban_sound_event_detection/detect_sound_event.py)
- [urban_sound_metrics.json](file:///home/sericsoft/Desktop/pytorch_100_projects/91_urban_sound_event_detection/outputs/urban_sound_metrics.json)
- [urban_sound_report.png](file:///home/sericsoft/Desktop/pytorch_100_projects/91_urban_sound_event_detection/outputs/urban_sound_report.png)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
