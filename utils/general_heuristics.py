from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class HeuristicResult:
    score: float  # 0..1 AI-likeness
    metrics: Dict[str, Any]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def estimate_text_artifact_score(image: np.ndarray) -> HeuristicResult:
    """Heuristic proxy for text-like artifact issues without OCR.

    Uses MSER to detect text-like regions and checks stroke/edge irregularity.
    High score indicates suspicious text-like patterns (common in AI images).
    """
    import cv2

    if image is None:
        return HeuristicResult(score=0.0, metrics={"error": "no_image"})

    img = image
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    else:
        gray = img

    # Normalize
    gray = cv2.equalizeHist(gray.astype(np.uint8))

    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
    except Exception as e:
        return HeuristicResult(score=0.0, metrics={"error": str(e)})

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return HeuristicResult(score=0.0, metrics={"error": "bad_shape"})

    # Count candidate text-like boxes
    boxes = []
    for p in regions[:400]:
        x, y, bw, bh = cv2.boundingRect(p)
        area = bw * bh
        if area < 60 or area > 0.1 * h * w:
            continue
        aspect = bw / float(bh + 1e-6)
        # Text-like regions are typically not extremely tall and not extremely wide.
        # Keep height conservative to avoid mistaking general texture for text.
        if 0.2 <= aspect <= 6.0 and 6 <= bh <= int(0.12 * h) and 6 <= bw <= int(0.6 * w):
            boxes.append((x, y, bw, bh))

    # Edge-based irregularity: AI text often looks "wobbly" with inconsistent stroke widths.
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    box_count = int(len(boxes))
    box_density = box_count / float(max(1, (h * w) / 10_000))  # per 10k px

    # Gating: if we don't have enough text-like regions, treat as "no text".
    if box_count < 10:
        return HeuristicResult(
            score=0.0,
            metrics={
                "mser_box_count": box_count,
                "mser_box_density": float(box_density),
                "edge_density": float(edge_density),
                "note": "insufficient_text_like_regions",
            },
        )

    # Text is usually localized to a band/area, not spread across the whole frame.
    xs = np.array([x for x, _, _, _ in boxes], dtype=float)
    ys = np.array([y for _, y, _, _ in boxes], dtype=float)
    x2s = np.array([x + bw for x, _, bw, _ in boxes], dtype=float)
    y2s = np.array([y + bh for _, y, _, bh in boxes], dtype=float)
    span_x = float((np.max(x2s) - np.min(xs)) / (w + 1e-6))
    span_y = float((np.max(y2s) - np.min(ys)) / (h + 1e-6))

    # If regions cover a large fraction of the image height, it's likely texture, not text.
    if span_y > 0.35:
        return HeuristicResult(
            score=0.0,
            metrics={
                "mser_box_count": box_count,
                "mser_box_density": float(box_density),
                "edge_density": float(edge_density),
                "span_x": float(span_x),
                "span_y": float(span_y),
                "note": "regions_spread_across_height",
            },
        )

    heights = np.array([bh for _, _, _, bh in boxes], dtype=float)
    h_mean = float(np.mean(heights)) if heights.size else 0.0
    h_std = float(np.std(heights)) if heights.size else 0.0
    h_cv = float(h_std / (h_mean + 1e-6)) if h_mean > 0 else 999.0

    # Score: require both density and shape consistency. If heights vary wildly, down-weight.
    density_term = min(0.55, box_density / 18.0)
    edge_term = min(0.25, max(0.0, (edge_density - 0.04) / 0.18))
    consistency_term = max(0.0, min(0.20, 0.20 * (1.0 - min(1.0, h_cv / 1.0))))

    score = density_term + edge_term + consistency_term
    score = float(max(0.0, min(0.9, score)))

    return HeuristicResult(
        score=score,
        metrics={
            "mser_box_count": box_count,
            "mser_box_density": float(box_density),
            "edge_density": float(edge_density),
            "height_cv": float(h_cv),
            "span_x": float(span_x),
            "span_y": float(span_y),
        },
    )


def estimate_geometry_inconsistency_score(image: np.ndarray) -> HeuristicResult:
    """General geometry inconsistency heuristic (no object detectors required).

    Uses line detection and checks whether strong lines exhibit unnatural curvature
    or excessive fragmentation (often caused by AI warping artifacts).
    """
    import cv2

    if image is None:
        return HeuristicResult(score=0.0, metrics={"error": "no_image"})

    img = image
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    else:
        gray = img

    gray = gray.astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 140)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=8)
    line_count = 0 if lines is None else int(len(lines))

    # Fragmentation: many short segments can indicate inconsistent geometry.
    frag = 0.0
    if lines is not None and len(lines) > 0:
        lengths = []
        for line in lines[:300]:
            x1, y1, x2, y2 = line[0]
            lengths.append(np.hypot(x2 - x1, y2 - y1))
        lengths = np.array(lengths, dtype=float)
        if lengths.size > 0:
            short_ratio = float(np.mean(lengths < 70.0))
            frag = short_ratio
    else:
        frag = 0.0

    edge_density = float(edges.mean() / 255.0)

    score = 0.0
    score += min(0.55, frag * 0.75)
    score += min(0.45, max(0.0, (edge_density - 0.05) / 0.20))
    score = float(max(0.0, min(1.0, score)))

    return HeuristicResult(
        score=score,
        metrics={
            "line_count": int(line_count),
            "fragmentation": float(frag),
            "edge_density": float(edge_density),
        },
    )
