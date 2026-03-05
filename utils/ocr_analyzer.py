from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class OCRTextResult:
    is_suspicious: bool
    confidence_score: float  # 0..1 AI-likeness of text rendering (NOT AI probability)
    text_detected: bool
    indicators: List[str]
    metrics: Dict[str, Any]


_READER = None


def _get_easyocr_reader():
    global _READER
    if _READER is not None:
        return _READER

    try:
        import easyocr  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "easyocr is not installed. Install with: pip install easyocr"
        ) from e

    # Force CPU; cache models locally under checkpoints/
    _READER = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory="checkpoints/ocr/models",
        user_network_directory="checkpoints/ocr/user_network",
        verbose=False,
    )
    return _READER


def analyze_text_ocr(image: Any, min_confidence: float = 0.2) -> OCRTextResult:
    """OCR-based text artifact analysis.

    Returns a score that estimates whether detected text looks *artifacty/unreliable*.

    - If no text is detected -> confidence_score=0.0
    - If text is detected:
        - low OCR confidence / many tiny fragments -> higher suspicion
        - high confidence / coherent tokens -> lower suspicion

    This is meant to drive the "Text / Symbol Artifacts" category, not final AI probability.
    """

    if image is None:
        return OCRTextResult(
            is_suspicious=False,
            confidence_score=0.0,
            text_detected=False,
            indicators=["No image provided"],
            metrics={},
        )

    img = image
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img

    reader = _get_easyocr_reader()

    try:
        # detail=1 => (bbox, text, conf)
        results = reader.readtext(img_rgb, detail=1)
    except Exception as e:
        return OCRTextResult(
            is_suspicious=False,
            confidence_score=0.0,
            text_detected=False,
            indicators=[f"OCR failed: {e}"],
            metrics={"error": str(e)},
        )

    # Filter by confidence and minimal token length
    tokens = []
    confs = []
    lengths = []
    for r in results:
        if not isinstance(r, (list, tuple)) or len(r) < 3:
            continue
        _bbox, text, conf = r[0], str(r[1]), float(r[2])
        text_norm = "".join(ch for ch in text if not ch.isspace())
        if len(text_norm) == 0:
            continue
        if conf < min_confidence:
            continue
        tokens.append(text)
        confs.append(conf)
        lengths.append(len(text_norm))

    if len(tokens) == 0:
        return OCRTextResult(
            is_suspicious=False,
            confidence_score=0.0,
            text_detected=False,
            indicators=["No text detected"],
            metrics={"raw_box_count": int(len(results))},
        )

    confs_arr = np.array(confs, dtype=float)
    lengths_arr = np.array(lengths, dtype=float)

    mean_conf = float(np.mean(confs_arr))
    low_conf_ratio = float(np.mean(confs_arr < 0.55))
    tiny_ratio = float(np.mean(lengths_arr <= 2))

    # Artifact score: high if confidence is low or many fragments
    # Keep conservative so real images with text don't get punished too harshly.
    score = 0.0
    score += 0.70 * max(0.0, (0.75 - mean_conf) / 0.75)  # mean_conf 0.75->0, 0->1
    score += 0.20 * low_conf_ratio
    score += 0.10 * tiny_ratio
    score = float(np.clip(score, 0.0, 1.0))

    is_suspicious = bool(score >= 0.6)

    indicators: List[str] = []
    indicators.append(f"Text detected ({len(tokens)} regions)")
    indicators.append(f"Mean OCR confidence: {mean_conf:.2f}")
    if low_conf_ratio > 0.5:
        indicators.append("Many low-confidence text regions")
    if tiny_ratio > 0.5:
        indicators.append("Many tiny/fractured text tokens")

    metrics: Dict[str, Any] = {
        "text_regions": int(len(tokens)),
        "mean_conf": mean_conf,
        "low_conf_ratio": low_conf_ratio,
        "tiny_token_ratio": tiny_ratio,
        "sample_tokens": tokens[:6],
    }

    return OCRTextResult(
        is_suspicious=is_suspicious,
        confidence_score=score,
        text_detected=True,
        indicators=indicators,
        metrics=metrics,
    )
