# AI-Generated Video Detector

A comprehensive multimodal deepfake detection system that combines deep learning with physics-based forensic analysis to detect AI-generated content across images, videos, and audio.

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Analysis Modules](#analysis-modules)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)

---

## Overview

This system detects AI-generated content (deepfakes) using a **multi-layered approach**:

1. **Visual Analysis** - Deep learning models (EfficientNet + CLIP and optional Transformers-based detector) for visual artifact detection
2. **Forensic Analysis** - Physics-based signal detection (PRNU, CFA, frequency, noise)
3. **Temporal Analysis** - Video-specific motion and identity consistency
4. **Audio Analysis** - Speech phase coherence and microstructure patterns
5. **OCR-based Text Analysis** - Optional OCR (EasyOCR) used to stabilize the Text/Symbol artifact category
6. **Calibrated Fusion** - Bayesian/Logistic combination of all signals

The system provides **unified verdicts** with confidence scores and uncertainty quantification.

---

## Tech Stack

### Core Deep Learning

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Visual Model** | EfficientNetV2 | Face/image classification |
| **Vision-Language** | CLIP (OpenAI) | Semantic artifact detection |
| **Scene Understanding** | BLIP-2 (Salesforce) | Contextual scene analysis |
| **Audio Model** | Custom Spectral CNN | Audio deepfake detection |
| **Face Detection** | MTCNN (facenet-pytorch) | Face extraction |

### Forensic Analysis

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Wavelet Processing** | PyWavelets | PRNU noise extraction |
| **Signal Processing** | SciPy | FFT, correlation, filtering |
| **Image Processing** | OpenCV | Frame extraction, optical flow |
| **Audio Processing** | pydub | Audio loading (mono conversion) |
| **OCR** | EasyOCR | Text detection + confidence scoring |

### Web Interface

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Gradio | Web UI |
| **Backend** | Gradio API | REST endpoints |
| **Model Serving** | PyTorch | Inference runtime |

### Dependencies

```
PyTorch >= 2.0
EfficientNet (timm)
CLIP (openai-clip)
BLIP-2 (transformers)
PyWavelets >= 1.3.0
SciPy >= 1.10
OpenCV >= 4.7
pydub >= 0.25
easyocr >= 1.7
Gradio >= 4.0
NumPy, Pillow, mutagen
```

---

## Capabilities

### Image Analysis

| Analysis | What It Detects | Method |
|----------|----------------|--------|
| **Visual Artifacts** | Skin texture, blending, asymmetry | EfficientNet + CLIP |
| **PRNU Fingerprint** | Camera sensor signature absence | Wavelet denoising + correlation |
| **CFA/Demosaicing** | Bayer filter pattern anomalies | Color channel correlation |
| **Frequency Domain** | GAN/diffusion spectral artifacts | FFT analysis |
| **Noise Residual** | Synthetic noise uniformity | Wavelet noise extraction |
| **Metadata** | AI software signatures, missing EXIF | EXIF/container parsing |
| **Text / Symbol Artifacts** | OCR confidence instability, fractured text tokens | EasyOCR (fallback: heuristics) |

### Video Analysis

| Analysis | What It Detects | Method |
|----------|----------------|--------|
| **Temporal Consistency** | Frame-to-frame coherence | Optical flow (Farneback) |
| **Identity Stability** | Face drift across frames | Feature tracking (ORB) |
| **Motion Blur** | Physics inconsistencies | Laplacian variance |
| **Temporal Noise** | Noise pattern consistency | Inter-frame correlation |
| **Frame Transitions** | Unnatural cuts/editing | Scene change detection |

### Audio Analysis

| Analysis | What It Detects | Method |
|----------|----------------|--------|
| **Spectral Patterns** | Vocoder artifacts | Spectrogram CNN |
| **Phase Coherence** | Unnatural phase relationships | STFT phase analysis |
| **Prosody** | Over-regular rhythm/intonation | Pitch + energy contour |
| **Breath/Pause** | Missing natural speech patterns | Low-energy detection |
| **Formants** | Vocal tract discontinuities | LPC analysis |
| **Microstructure** | Timing/amplitude variations | Zero-crossing analysis |

### Fusion System

| Feature | Description |
|---------|-------------|
| **Bayesian Fusion** | Probabilistic evidence combination |
| **Logistic Calibration** | Sigmoid-based probability mapping |
| **Consensus Detection** | Bonus when forensics agree |
| **Uncertainty Quantification** | Confidence intervals |
| **Weighted Combination** | Analyzer reliability weights |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT MEDIA                               │
│                   (Image / Video / Audio)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEDIA TYPE ROUTER                             │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  IMAGE PATH   │       │  VIDEO PATH   │       │  AUDIO PATH   │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Visual Model  │       │ Visual Model  │       │ Spectral Model│
│ (EfficientNet │       │ (Frame-by-    │       │ (CNN on       │
│  + CLIP)      │       │  frame)       │       │  spectrogram) │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   Metadata    │       │   Metadata    │       │   Metadata    │
│   Analyzer    │       │   Analyzer    │       │   Analyzer    │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   Frequency   │       │   Temporal    │       │  Audio Phase  │
│   Analyzer    │       │   Analyzer    │       │   Analyzer    │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│    PRNU       │       │    Noise      │       │    Noise      │
│   Analyzer    │       │   Analyzer    │       │   Analyzer    │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│     CFA       │       │               │       │               │
│   Analyzer    │       │               │       │               │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│    Noise      │       │               │       │               │
│   Analyzer    │       │               │       │               │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CALIBRATED FUSION ENGINE                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Bayesian Fusion │  │ Logistic Fusion │  │ Consensus Bonus │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL VERDICT                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ AI_GENERATED│  │    REAL     │  │      UNCERTAIN          │  │
│  │   (>55%)    │  │   (<40%)    │  │      (40-55%)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  + Confidence Score + Uncertainty + Key Indicators              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Analysis Modules

### 1. Visual Analyzer (`deepfakes_image_predict`)

**Location:** `inference_2.py`

**Models:**
- EfficientNetV2 (image classification)
- CLIP (vision-language artifact detection)

**Process:**
1. Face detection via MTCNN
2. Face crop with margin
3. EfficientNet classification (real/fake)
4. CLIP semantic matching against artifact descriptions
5. Ensemble combination (60% EfficientNet, 40% CLIP)

**Output:** Verdict string with confidence and indicators

---

### 2. Metadata Analyzer (`utils/metadata_analyzer.py`)

**Detects:**
- Missing camera EXIF data
- AI software signatures (Photoshop AI, Stable Diffusion, DALL-E, etc.)
- Inconsistent timestamps
- Container anomalies
- Audio encoder metadata

**Confidence Score:** Based on number of suspicious findings

---

### 3. Frequency Analyzer (`utils/frequency_analyzer.py`)

**Detects:**
- Low high-frequency content (smoothed AI images)
- Spectral flatness anomalies
- Grid-like artifacts (GAN checkerboard patterns)
- Frequency ringing (diffusion artifacts)

**Method:** FFT analysis with statistical comparison to natural image spectra

---

### 4. PRNU Analyzer (`utils/prnu_analyzer.py`)

**Detects:**
- Absence of camera sensor fingerprint
- Unnatural noise spectrum
- Excessive high-frequency noise (GAN artifacts)
- Block artifacts (manipulation)

**Method:**
1. Wavelet denoising (Daubechies)
2. Noise residual extraction
3. Block-wise correlation analysis
4. Spectral analysis of noise

---

### 5. CFA Analyzer (`utils/cfa_analyzer.py`)

**Detects:**
- Presence/absence of Bayer filter pattern
- Demosaicing artifacts
- Abnormal color channel correlations
- Periodic color structure

**Method:**
1. Color channel separation
2. Periodic pattern detection
3. Cross-channel correlation
4. Bayer pattern identification

---

### 6. Noise Analyzer (`utils/noise_analyzer.py`)

**Detects:**
- Noise uniformity (AI has uniform noise)
- Noise naturalness (real cameras have non-Gaussian noise)
- Synthetic noise patterns
- Temporal noise consistency (video)

**Method:**
1. Wavelet noise extraction
2. Statistical analysis (kurtosis, skewness)
3. Spectral analysis
4. Cross-channel comparison

---

### 7. Temporal Analyzer (`utils/temporal_analyzer.py`)

**Detects:**
- Optical flow inconsistencies
- Identity drift (face changes across frames)
- Motion blur anomalies
- Temporal noise patterns
- Unnatural frame transitions

**Method:**
1. Frame extraction
2. Optical flow (Farneback)
3. Feature tracking (ORB)
4. Motion blur analysis (Laplacian variance)
5. Inter-frame correlation

---

### 8. Audio Phase Analyzer (`utils/audio_phase_analyzer.py`)

**Detects:**
- Phase coherence anomalies
- Missing breath/pause patterns
- Over-regular prosody
- Formant discontinuities
- Microstructure regularity

**Method:**
1. STFT phase extraction
2. Energy-based pause detection
3. Pitch contour analysis
4. LPC formant estimation
5. Zero-crossing microstructure

---

### 9. Fusion Analyzer (`utils/fusion_analyzer.py`)

**Weights:**

| Analyzer | Weight | Reason |
|----------|--------|--------|
| PRNU | 0.25 | Physics-based, reliable |
| CFA | 0.20 | Physics-based, reliable |
| Frequency | 0.18 | Good GAN detection |
| Noise | 0.15 | Moderate reliability |
| Visual | 0.15 | Deep learning baseline |
| Metadata | 0.07 | Can be spoofed |

**Fusion Methods:**
1. **Bayesian Fusion** - Log-odds combination with prior
2. **Logistic Fusion** - Weighted sum through sigmoid
3. **Consensus Bonus** - Small probability bonus when multiple forensics agree (kept conservative to reduce false positives)

**Verdict Thresholds:**
- AI_GENERATED: > 70%
- REAL: < 30%
- UNCERTAIN: 30-70%

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install efficientnet-pytorch timm
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install transformers accelerate
pip install facenet-pytorch

# Forensic analysis
pip install PyWavelets scipy opencv-python
pip install pydub

# Optional OCR (text/symbol artifact category)
pip install easyocr

# Metadata extraction
pip install pillow mutagen ffmpeg-python

# Web interface
pip install gradio
```

### Setup

```bash
# Clone repository
git clone https://github.com/Alien2002/AI-Generated-Video-Detector.git
cd AI-Generated-Video-Detector

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

---

## Usage

### Web Interface

```bash
python app.py
# Open http://localhost:7860
```

### Programmatic Usage

```python
from inference_2 import (
    deepfakes_image_predict_with_metadata,
    deepfakes_video_predict_with_metadata,
    deepfakes_audio_predict_with_metadata
)

# Image analysis
result = deepfakes_image_predict_with_metadata("path/to/image.jpg")
print(result)

# Video analysis
result = deepfakes_video_predict_with_metadata("path/to/video.mp4")
print(result)

# Audio analysis
result = deepfakes_audio_predict_with_metadata("path/to/audio.wav")
print(result)
```

### Individual Analyzers

```python
from utils.frequency_analyzer import analyze_frequency
from utils.prnu_analyzer import analyze_prnu
from utils.cfa_analyzer import analyze_cfa
from utils.noise_analyzer import analyze_noise
from utils.temporal_analyzer import analyze_temporal
from utils.audio_phase_analyzer import analyze_audio_phase
from utils.fusion_analyzer import fuse_image_analysis

import numpy as np
from PIL import Image

# Load image
img = np.array(Image.open("image.jpg"))

# Run individual analyses
freq_result = analyze_frequency(img)
prnu_result = analyze_prnu(img)
cfa_result = analyze_cfa(img)
noise_result = analyze_noise(img, "image")

# Fuse results
fusion = fuse_image_analysis(
    frequency_result=freq_result,
    prnu_result=prnu_result,
    cfa_result=cfa_result,
    noise_result=noise_result
)

print(f"Verdict: {fusion.final_verdict}")
print(f"Confidence: {fusion.confidence:.2%}")
print(f"AI Probability: {fusion.calibrated_probability:.2%}")
```

---

## API Reference

### Image Prediction

```python
deepfakes_image_predict_with_metadata(input_image_path) -> str
```

**Input:** Path to image file or numpy array

**Output:** Multi-line string containing:
- Visual verdict with confidence
- Metadata analysis (if path provided)
- Frequency analysis
- PRNU analysis
- CFA analysis
- Noise analysis
- Calibrated fusion verdict

---

### Video Prediction

```python
deepfakes_video_predict_with_metadata(input_video) -> str
```

**Input:** Path to video file

**Output:** Multi-line string containing:
- Visual verdict (frame-by-frame)
- Metadata analysis
- Noise analysis
- Temporal consistency analysis
- Calibrated fusion verdict

---

### Audio Prediction

```python
deepfakes_audio_predict_with_metadata(input_audio_path) -> str
```

**Input:** Path to audio file or (audio_array, sample_rate) tuple

**Output:** Multi-line string containing:
- Spectral verdict
- Metadata analysis
- Noise analysis
- Audio phase analysis
- Calibrated fusion verdict

---

## Configuration

### Fusion Weights

Edit `utils/fusion_analyzer.py`:

```python
self.analyzer_weights = {
    'prnu': 0.25,
    'cfa': 0.20,
    'frequency': 0.18,
    'noise_image': 0.15,
    'visual': 0.15,
    'metadata': 0.07,
}
```

### Verdict Thresholds

Edit `utils/fusion_analyzer.py`:

```python
if final_probability > 0.70:  # AI_GENERATED threshold
    verdict = "AI_GENERATED"
elif final_probability < 0.30:  # REAL threshold
    verdict = "REAL"
else:
    verdict = "UNCERTAIN"
```

### Calibration Parameters

Edit `utils/fusion_analyzer.py`:

```python
self.calibration_params = {
    # Intercepts are centered so score=0.5 maps close to probability=0.5
    # (intercept ≈ -slope * 0.5)
    'prnu': {'slope': 6.0, 'intercept': -3.0},
    'frequency': {'slope': 5.0, 'intercept': -2.5},
    'noise': {'slope': 5.0, 'intercept': -2.5},
    'visual': {'slope': 4.0, 'intercept': -2.0},
}
```

Higher slope = more sensitive, lower intercept = lower baseline probability.

---

## Output Example

```
VERDICT: Likely FAKE (78.35% confidence)

⚠️  Fake indicators: overly smooth porcelain skin | waxy artificial skin
✓  Real indicators: natural skin pores and fine texture

Summary: Synthetic artifacts dominate. Some natural elements present but outweighed by artifacts.

🔬 FREQUENCY ANALYSIS:
Confidence: 55.0%
Indicators: Low high-frequency content, Unusual spectral flatness, Grid-like artifacts

📷 SENSOR FINGERPRINT (PRNU):
Confidence: 45.0%
PRNU Present: Yes
Indicators: Unnatural noise spectrum, Excessive high-frequency noise

🔊 NOISE RESIDUAL ANALYSIS:
Confidence: 70.0%
Uniformity: 0.997 (high = synthetic)
Indicators: Highly uniform noise, Synthetic noise patterns detected

🧮 CALIBRATED FUSION:
Verdict: AI_GENERATED
Confidence: 68.2%
Uncertainty: ±8.5%
AI Probability: 62.3%
Forensic Consensus: 3/3 forensic signals indicate AI
```

---

## License

MIT License

---

## Authors

- **Original Project:** Alien2002
- **Forensic Analysis Extensions:** Added PRNU, CFA, Frequency, Noise, Temporal, Audio Phase, and Fusion analyzers

---

## References

- [PRNU-based Camera Identification](https://ieeexplore.ieee.org/document/1643822)
- [CFA/Demosaicing Detection](https://ieeexplore.ieee.org/document/1651968)
- [GAN Frequency Artifacts](https://arxiv.org/abs/1902.04356)
- [Audio Deepfake Detection](https://arxiv.org/abs/2104.04069)
- [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
