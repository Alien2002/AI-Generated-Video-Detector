import numpy as np
from typing import List, Tuple, Optional

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


class TransformersVisualDetector:
    def __init__(
        self,
        model_id: str = "prithivMLmods/Deep-Fake-Detector-v2-Model",
        cache_dir: str = "checkpoints/hf_cache",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.deepfake_label_id = None
        if getattr(cfg, "label2id", None) and "Deepfake" in cfg.label2id:
            self.deepfake_label_id = int(cfg.label2id["Deepfake"])
        else:
            for k, v in getattr(cfg, "id2label", {}).items():
                if str(v).lower() in {"deepfake", "fake", "ai", "ai_generated"}:
                    self.deepfake_label_id = int(k)
                    break
        if self.deepfake_label_id is None:
            self.deepfake_label_id = 1

    def _to_pil_or_np(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _sample_patches(
        self,
        image: np.ndarray,
        patch_size: int = 224,
        num_random: int = 6,
    ) -> List[np.ndarray]:
        h, w = image.shape[:2]
        if h < patch_size or w < patch_size:
            import cv2

            scale = patch_size / float(min(h, w))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        patches: List[np.ndarray] = []
        coords: List[Tuple[int, int]] = []

        coords.extend(
            [
                (0, 0),
                (0, w - patch_size),
                (h - patch_size, 0),
                (h - patch_size, w - patch_size),
                ((h - patch_size) // 2, (w - patch_size) // 2),
            ]
        )

        rng = np.random.default_rng()
        for _ in range(max(0, num_random)):
            y = int(rng.integers(0, h - patch_size + 1))
            x = int(rng.integers(0, w - patch_size + 1))
            coords.append((y, x))

        for y, x in coords:
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

        return patches

    @torch.no_grad()
    def predict_deepfake_probability(
        self,
        image: np.ndarray,
        patch_size: int = 224,
        num_random_patches: int = 6,
    ) -> float:
        image = self._to_pil_or_np(image)
        patches = self._sample_patches(image, patch_size=patch_size, num_random=num_random_patches)

        inputs = self.processor(images=patches, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        deepfake_probs = probs[:, self.deepfake_label_id]

        mean_p = float(deepfake_probs.mean().item())
        max_p = float(deepfake_probs.max().item())
        fused = 0.7 * mean_p + 0.3 * max_p
        return float(max(0.0, min(1.0, fused)))
