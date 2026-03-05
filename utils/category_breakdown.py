from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.general_heuristics import estimate_geometry_inconsistency_score, estimate_text_artifact_score
from utils.ocr_analyzer import OCRTextResult


@dataclass
class CategoryItem:
    name: str
    score: float  # 0..1 AI-likeness
    description: str
    metrics: Dict[str, Any]


def build_image_categories(
    *,
    image: Optional[Any] = None,
    ocr_result: Optional[OCRTextResult] = None,
    frequency_result: Optional[Any] = None,
    noise_result: Optional[Any] = None,
    prnu_result: Optional[Any] = None,
    cfa_result: Optional[Any] = None,
    metadata_result: Optional[Any] = None,
    visual_probability: Optional[float] = None,
) -> List[CategoryItem]:
    """General-image categories (no face dependency) derived from analyzers.

    Scores are heuristic and should be calibrated later.
    """

    cats: List[CategoryItem] = []

    # Texture / Over-smoothing
    tex_score = 0.0
    tex_metrics: Dict[str, Any] = {}
    if frequency_result is not None:
        # Cap frequency contribution to 0.4 to avoid single analyzer dominating
        freq_conf = float(getattr(frequency_result, "confidence_score", 0.0))
        tex_score = max(tex_score, min(0.4, freq_conf))
        tex_metrics["frequency_confidence"] = freq_conf
    if noise_result is not None:
        # Use noise analyzer's own confidence (incorporates uniformity, naturalness, synthetic patterns)
        noise_conf = float(getattr(noise_result, "confidence_score", 0.0))
        tex_metrics["noise_confidence"] = noise_conf
        tex_metrics["noise_uniformity"] = float(getattr(noise_result, "noise_uniformity", 0.0))
        # Scale down noise confidence for texture category (it's more about noise realism)
        tex_score = max(tex_score, min(0.5, noise_conf * 0.7))
    cats.append(
        CategoryItem(
            name="Texture / Over-smoothing",
            score=float(max(0.0, min(1.0, tex_score))),
            description=(
                "Checks for overly smooth textures and reduced natural micro-detail commonly seen in AI-generated images."
            ),
            metrics=tex_metrics,
        )
    )

    # Repetition / Tiling / Background artifacts (proxy via frequency/grid indicators)
    rep_score = 0.0
    rep_metrics: Dict[str, Any] = {}
    if frequency_result is not None:
        inds = getattr(frequency_result, "indicators", []) or []
        rep_metrics["frequency_indicators"] = inds[:3]
        rep_score = 0.65 if any("grid" in str(i).lower() for i in inds) else 0.25
    cats.append(
        CategoryItem(
            name="Repetition / Background Artifacts",
            score=float(max(0.0, min(1.0, rep_score))),
            description=(
                "Looks for repeating patterns, grid artifacts, and unnatural background coherence that often appear in synthetic images."
            ),
            metrics=rep_metrics,
        )
    )

    # Text artifacts (prefer OCR-based, fallback to OCR-free proxy)
    if ocr_result is not None:
        cats.append(
            CategoryItem(
                name="Text / Symbol Artifacts",
                score=float(max(0.0, min(1.0, float(ocr_result.confidence_score)))),
                description=(
                    "OCR-based check for unreliable/fractured text rendering. If no text is detected, this score stays low."
                ),
                metrics=ocr_result.metrics,
            )
        )
    elif image is not None:
        try:
            txt = estimate_text_artifact_score(image)
            cats.append(
                CategoryItem(
                    name="Text / Symbol Artifacts",
                    score=float(max(0.0, min(1.0, txt.score))),
                    description=(
                        "OCR-free proxy based on text-like regions and edge structure. Conservative fallback when OCR is unavailable."
                    ),
                    metrics=txt.metrics,
                )
            )
        except Exception:
            pass

    # Geometry consistency (general)
    if image is not None:
        try:
            geo = estimate_geometry_inconsistency_score(image)
            cats.append(
                CategoryItem(
                    name="Geometry Consistency",
                    score=float(max(0.0, min(1.0, geo.score))),
                    description=(
                        "Checks for fragmented/warped straight-line structure that can arise from AI synthesis artifacts."
                    ),
                    metrics=geo.metrics,
                )
            )
        except Exception:
            pass

    # Noise realism
    noise_score = 0.0
    noise_metrics: Dict[str, Any] = {}
    if noise_result is not None:
        noise_score = float(getattr(noise_result, "confidence_score", 0.0))
        noise_metrics["confidence"] = noise_score
        noise_metrics["noise_uniformity"] = float(getattr(noise_result, "noise_uniformity", 0.0))
    cats.append(
        CategoryItem(
            name="Noise Realism",
            score=float(max(0.0, min(1.0, noise_score))),
            description=(
                "Evaluates whether noise patterns look like real camera noise or unusually uniform/synthetic residuals."
            ),
            metrics=noise_metrics,
        )
    )

    # Camera pipeline evidence (PRNU + CFA)
    pipe_score = 0.5
    pipe_metrics: Dict[str, Any] = {}
    if prnu_result is not None:
        pipe_metrics["prnu_present"] = bool(getattr(prnu_result, "prnu_present", False))
        pipe_metrics["prnu_confidence"] = float(getattr(prnu_result, "confidence_score", 0.0))
        if not bool(getattr(prnu_result, "prnu_present", False)):
            pipe_score = 0.75
    if cfa_result is not None:
        pipe_metrics["cfa_detected"] = bool(getattr(cfa_result, "cfa_detected", False))
        pipe_metrics["cfa_confidence"] = float(getattr(cfa_result, "confidence_score", 0.0))
        if not bool(getattr(cfa_result, "cfa_detected", False)):
            pipe_score = max(pipe_score, 0.75)
    cats.append(
        CategoryItem(
            name="Camera Pipeline Evidence",
            score=float(max(0.0, min(1.0, pipe_score))),
            description=(
                "Checks for evidence of a real camera imaging pipeline (sensor fingerprint/PRNU and demosaicing/CFA artifacts)."
            ),
            metrics=pipe_metrics,
        )
    )

    # Metadata trust (low weight but explainable)
    meta_score = 0.0
    meta_metrics: Dict[str, Any] = {}
    if metadata_result is not None:
        meta_score = float(getattr(metadata_result, "confidence_score", 0.0))
        meta_metrics["indicators"] = (getattr(metadata_result, "indicators", []) or [])[:3]
    cats.append(
        CategoryItem(
            name="Metadata / Provenance",
            score=float(max(0.0, min(1.0, meta_score))),
            description=(
                "Reviews file metadata for provenance clues (missing camera tags, unusual software signatures). Metadata can be missing in legitimate cases."
            ),
            metrics=meta_metrics,
        )
    )

    # Visual model score (pretrained classifier)
    if visual_probability is not None:
        cats.append(
            CategoryItem(
                name="Learned Visual Detector",
                score=float(max(0.0, min(1.0, float(visual_probability)))),
                description=(
                    "A pretrained image classifier estimating the likelihood the image is synthetic. Patch sampling improves robustness."
                ),
                metrics={"visual_probability": float(visual_probability)},
            )
        )

    return cats


def categories_to_dict(categories: List[CategoryItem]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in categories:
        out.append({
            "name": c.name,
            "score": float(c.score),
            "description": c.description,
            "metrics": c.metrics,
        })
    return out
