from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import json
import numpy as np
from scipy.optimize import minimize


@dataclass
class PlattCalibrator:
    """Platt scaling: p = sigmoid(a * logit + b)."""

    a: float = 1.0
    b: float = 0.0

    def predict_proba(self, p: float) -> float:
        # Convert probability to logit, then apply calibration
        eps = 1e-6
        p = float(np.clip(p, eps, 1.0 - eps))
        logit = np.log(p / (1.0 - p))
        z = self.a * logit + self.b
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlattCalibrator":
        return PlattCalibrator(a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)))


def fit_platt_scaling(raw_probs: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Fit Platt scaling parameters on probabilities.

    raw_probs: model probabilities (0..1)
    labels: 0/1 ground truth (1 = AI)
    """

    raw_probs = np.asarray(raw_probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    eps = 1e-6
    p = np.clip(raw_probs, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))

    def nll(params):
        a, b = params
        z = a * logits + b
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = np.clip(pred, eps, 1.0 - eps)
        return float(-np.mean(labels * np.log(pred) + (1.0 - labels) * np.log(1.0 - pred)))

    res = minimize(nll, x0=np.array([1.0, 0.0], dtype=float), method="L-BFGS-B")
    a, b = res.x
    return PlattCalibrator(a=float(a), b=float(b))


def load_calibrator(path: str) -> Optional[PlattCalibrator]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PlattCalibrator.from_dict(data)
    except Exception:
        return None


def save_calibrator(path: str, calibrator: PlattCalibrator) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibrator.to_dict(), f, indent=2)
