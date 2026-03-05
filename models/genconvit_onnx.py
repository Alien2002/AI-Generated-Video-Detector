"""
GenConViT ONNX Inference Wrapper
Loads and runs GenConViT ONNX models for deepfake video detection.
"""
import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple, Optional


class GenConViTONNX:
    """
    GenConViT Deepfake Detector using ONNX Runtime.
    Supports ED (Autoencoder), VAE, or ensemble of both.
    """
    
    def __init__(self, 
                 ed_path: Optional[str] = None, 
                 vae_path: Optional[str] = None,
                 providers: Optional[list] = None):
        """
        Initialize GenConViT ONNX models.
        
        Args:
            ed_path: Path to GenConViT ED ONNX model
            vae_path: Path to GenConViT VAE ONNX model
            providers: ONNX Runtime execution providers (default: CUDA if available, else CPU)
        """
        self.sessions = {}
        
        # Default providers: try CUDA first, fall back to CPU
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load ED model
        if ed_path and os.path.exists(ed_path):
            self.sessions['ed'] = ort.InferenceSession(ed_path, providers=providers)
            print(f"Loaded GenConViT ED model from {ed_path}")
        
        # Load VAE model
        if vae_path and os.path.exists(vae_path):
            self.sessions['vae'] = ort.InferenceSession(vae_path, providers=providers)
            print(f"Loaded GenConViT VAE model from {vae_path}")
        
        if not self.sessions:
            raise FileNotFoundError("No valid model paths provided")
        
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input.
        
        Args:
            face_img: Face image (H, W, 3) in RGB format, values 0-255
        
        Returns:
            Preprocessed tensor (1, 3, 224, 224) as float32
        """
        # Resize to 224x224 (GenConViT input size)
        face = cv2.resize(face_img, (224, 224))
        
        # Normalize to 0-1
        face = face.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        face = (face - self.mean) / self.std
        
        # HWC to CHW format
        face = face.transpose(2, 0, 1)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face.astype(np.float32)
    
    def predict(self, face_tensor: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on preprocessed face(s).
        
        Args:
            face_tensor: Preprocessed face tensor (N, 3, 224, 224) or (1, 3, 224, 224)
        
        Returns:
            Tuple of (prediction_class, confidence)
            - prediction_class: 0 = REAL, 1 = FAKE
            - confidence: Probability of the predicted class (0-1)
        """
        results = []
        
        for name, session in self.sessions.items():
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: face_tensor})
            
            # Apply sigmoid to get probabilities
            logits = outputs[0]
            probs = 1 / (1 + np.exp(-logits))  # Sigmoid
            results.append(probs[0])  # Remove batch dim
        
        # Ensemble: average probabilities if both models present
        if len(results) == 2:
            avg_probs = (results[0] + results[1]) / 2
        else:
            avg_probs = results[0]
        
        # Get prediction (0 = REAL, 1 = FAKE)
        pred_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_class])
        
        return pred_class, confidence
    
    def predict_video_frames(self, faces: list) -> dict:
        """
        Predict on multiple video frames and aggregate results.
        
        Args:
            faces: List of face images (H, W, 3) in RGB format
        
        Returns:
            Dictionary with prediction results
        """
        if not faces:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'is_fake': None,
                'frame_count': 0
            }
        
        predictions = []
        confidences = []
        
        for face in faces:
            face_tensor = self.preprocess(face)
            pred_class, conf = self.predict(face_tensor)
            predictions.append(pred_class)
            confidences.append(conf)
        
        # Majority vote for final prediction
        fake_count = sum(predictions)
        real_count = len(predictions) - fake_count
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Determine final prediction
        if fake_count > real_count:
            is_fake = True
            prediction = 'FAKE'
            confidence = avg_confidence
        else:
            is_fake = False
            prediction = 'REAL'
            confidence = avg_confidence
        
        return {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'is_fake': is_fake,
            'frame_count': len(faces),
            'fake_frames': fake_count,
            'real_frames': real_count
        }


# Convenience factory function
def load_genconvit_onnx(weights_dir: str = 'checkpoints/genconvit_weights',
                        use_ed: bool = True,
                        use_vae: bool = True) -> GenConViTONNX:
    """
    Load GenConViT ONNX models from directory.
    
    Args:
        weights_dir: Directory containing ONNX files
        use_ed: Load ED model
        use_vae: Load VAE model
    
    Returns:
        GenConViTONNX instance
    """
    ed_path = os.path.join(weights_dir, 'genconvit_ed_inference.onnx') if use_ed else None
    vae_path = os.path.join(weights_dir, 'genconvit_vae_inference.onnx') if use_vae else None
    
    return GenConViTONNX(ed_path=ed_path, vae_path=vae_path)
