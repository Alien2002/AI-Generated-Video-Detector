"""
Temporal Consistency Analyzer for AI-Generated Video Detection

Analyzes temporal coherence in video sequences to detect AI-generated content.

Key insight: AI-generated videos often fail temporal physics tests:
- Optical flow inconsistencies (unnatural motion patterns)
- Identity drift (face/feature changes across frames)
- Motion blur anomalies (missing or incorrect blur physics)
- Rolling shutter artifacts (real cameras have specific shutter effects)
- Temporal noise patterns (inconsistent noise across frames)

Real videos maintain consistent physics-based temporal relationships.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TemporalResult:
    """Result of temporal consistency analysis"""
    is_suspicious: bool
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    flow_consistency: float = 0.0
    identity_stability: float = 0.0
    motion_blur_score: float = 0.0
    temporal_noise_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class TemporalAnalyzer:
    """
    Analyzes temporal consistency in video sequences.
    
    Detects anomalies that indicate AI-generated or manipulated video content.
    """
    
    def __init__(self):
        # Thresholds
        self.flow_consistency_threshold = 0.7  # Below this = suspicious
        self.identity_stability_threshold = 0.8
        self.max_frames = 30  # Max frames to analyze
        
    def analyze_video(self, video_path: str) -> TemporalResult:
        """
        Analyze temporal consistency in a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            TemporalResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        try:
            # Extract frames
            frames = self._extract_frames(video_path)
            
            if len(frames) < 2:
                return TemporalResult(
                    is_suspicious=False,
                    confidence_score=0.0,
                    indicators=["Not enough frames for temporal analysis"],
                    details={'frame_count': len(frames)}
                )
            
            details['frame_count'] = len(frames)
            
            # 1. Optical Flow Consistency
            flow_result = self._analyze_optical_flow(frames)
            details['flow_consistency'] = flow_result['consistency']
            details['flow_anomalies'] = flow_result['anomalies']
            
            # 2. Identity Stability (face/feature tracking)
            identity_result = self._analyze_identity_stability(frames)
            details['identity_stability'] = identity_result['stability']
            details['identity_drift'] = identity_result['drift']
            
            # 3. Motion Blur Analysis
            blur_result = self._analyze_motion_blur(frames)
            details['motion_blur_score'] = blur_result['score']
            details['blur_anomalies'] = blur_result['anomalies']
            
            # 4. Temporal Noise Analysis
            noise_result = self._analyze_temporal_noise(frames)
            details['temporal_noise_score'] = noise_result['score']
            details['noise_inconsistency'] = noise_result['inconsistency']
            
            # 5. Frame Transition Analysis
            transition_result = self._analyze_frame_transitions(frames)
            details['transition_score'] = transition_result['score']
            details['sudden_changes'] = transition_result['sudden_changes']
            
            # Generate indicators and compute confidence
            
            # Check 1: Optical Flow
            if flow_result['consistency'] < self.flow_consistency_threshold:
                indicators.append(f"Low optical flow consistency ({flow_result['consistency']:.3f}) - unnatural motion")
                confidence += 0.25
            else:
                indicators.append(f"Optical flow consistency: {flow_result['consistency']:.3f}")
            
            if flow_result['anomalies'] > 3:
                indicators.append(f"Multiple flow anomalies detected ({flow_result['anomalies']})")
                confidence += 0.1
            
            # Check 2: Identity Stability
            if identity_result['stability'] < self.identity_stability_threshold:
                indicators.append(f"Identity drift detected ({identity_result['stability']:.3f}) - features changing across frames")
                confidence += 0.3
            else:
                indicators.append(f"Identity stability: {identity_result['stability']:.3f}")
            
            if identity_result['drift'] > 0.15:
                indicators.append(f"Significant feature drift ({identity_result['drift']:.3f})")
                confidence += 0.15
            
            # Check 3: Motion Blur
            if blur_result['anomalies'] > 2:
                indicators.append(f"Motion blur anomalies ({blur_result['anomalies']}) - physics inconsistency")
                confidence += 0.15
            
            if blur_result['score'] < 0.3:
                indicators.append("Missing expected motion blur in dynamic scenes")
                confidence += 0.1
            
            # Check 4: Temporal Noise
            if noise_result['inconsistency'] > 0.3:
                indicators.append(f"Inconsistent noise across frames ({noise_result['inconsistency']:.3f})")
                confidence += 0.15
            
            # Check 5: Frame Transitions
            if transition_result['sudden_changes'] > 3:
                indicators.append(f"Sudden frame transitions ({transition_result['sudden_changes']}) - possible editing")
                confidence += 0.1
            
            if transition_result['score'] < 0.5:
                indicators.append("Unnatural frame-to-frame transitions")
                confidence += 0.1
            
        except Exception as e:
            logger.error(f"Temporal analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return TemporalResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            flow_consistency=details.get('flow_consistency', 0),
            identity_stability=details.get('identity_stability', 0),
            motion_blur_score=details.get('motion_blur_score', 0),
            temporal_noise_score=details.get('temporal_noise_score', 0),
            details=details
        )
    
    def analyze_frames(self, frames: List[np.ndarray]) -> TemporalResult:
        """
        Analyze temporal consistency from a list of frames.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            TemporalResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        try:
            if len(frames) < 2:
                return TemporalResult(
                    is_suspicious=False,
                    confidence_score=0.0,
                    indicators=["Not enough frames for temporal analysis"],
                    details={'frame_count': len(frames)}
                )
            
            details['frame_count'] = len(frames)
            
            # Run all temporal analyses
            flow_result = self._analyze_optical_flow(frames)
            identity_result = self._analyze_identity_stability(frames)
            blur_result = self._analyze_motion_blur(frames)
            noise_result = self._analyze_temporal_noise(frames)
            transition_result = self._analyze_frame_transitions(frames)
            
            details.update({
                'flow_consistency': flow_result['consistency'],
                'identity_stability': identity_result['stability'],
                'motion_blur_score': blur_result['score'],
                'temporal_noise_score': noise_result['score'],
                'transition_score': transition_result['score']
            })
            
            # Generate indicators
            if flow_result['consistency'] < self.flow_consistency_threshold:
                indicators.append(f"Low optical flow consistency ({flow_result['consistency']:.3f})")
                confidence += 0.25
            
            if identity_result['stability'] < self.identity_stability_threshold:
                indicators.append(f"Identity drift detected ({identity_result['stability']:.3f})")
                confidence += 0.3
            
            if blur_result['anomalies'] > 2:
                indicators.append(f"Motion blur anomalies ({blur_result['anomalies']})")
                confidence += 0.15
            
            if noise_result['inconsistency'] > 0.3:
                indicators.append(f"Inconsistent temporal noise ({noise_result['inconsistency']:.3f})")
                confidence += 0.15
            
            if transition_result['sudden_changes'] > 3:
                indicators.append(f"Sudden frame transitions ({transition_result['sudden_changes']})")
                confidence += 0.1
            
        except Exception as e:
            logger.error(f"Temporal analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return TemporalResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            flow_consistency=details.get('flow_consistency', 0),
            identity_stability=details.get('identity_stability', 0),
            motion_blur_score=details.get('motion_blur_score', 0),
            temporal_noise_score=details.get('temporal_noise_score', 0),
            details=details
        )
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.max_frames)
        
        frame_idx = 0
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _analyze_optical_flow(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze optical flow consistency across frames.
        AI videos often have inconsistent or unnatural motion patterns.
        """
        if len(frames) < 2:
            return {'consistency': 1.0, 'anomalies': 0}
        
        consistencies = []
        anomalies = 0
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Analyze flow properties
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            direction = np.arctan2(flow[..., 1], flow[..., 0])
            
            mag_mean = np.mean(magnitude)
            
            # Check for flow coherence
            if mag_mean > 1:  # Only check if there's meaningful motion
                # Direction variance in moving regions
                moving_mask = magnitude > mag_mean
                if np.any(moving_mask):
                    dir_var = np.var(direction[moving_mask])
                    # High direction variance in moving regions = inconsistent
                    if dir_var > 2.0:  # radians^2
                        anomalies += 1
                
                # Check for flow smoothness (neighboring pixels should have similar flow)
                flow_grad_x = np.gradient(magnitude, axis=1)
                flow_grad_y = np.gradient(magnitude, axis=0)
                flow_roughness = np.mean(np.abs(flow_grad_x)) + np.mean(np.abs(flow_grad_y))
                
                # Normalize roughness by magnitude
                roughness_ratio = flow_roughness / (mag_mean + 1e-10)
                consistency = 1.0 / (1.0 + roughness_ratio)
                consistencies.append(consistency)
            
            prev_gray = curr_gray
        
        avg_consistency = np.mean(consistencies) if consistencies else 1.0
        
        return {
            'consistency': float(avg_consistency),
            'anomalies': int(anomalies)
        }
    
    def _analyze_identity_stability(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze identity/feature stability across frames.
        AI-generated faces/features may drift or change over time.
        """
        if len(frames) < 2:
            return {'stability': 1.0, 'drift': 0.0}
        
        # Use feature-based approach
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        stabilities = []
        drifts = []
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)
            
            if prev_des is not None and curr_des is not None and len(prev_des) > 10 and len(curr_des) > 10:
                # Match features
                matches = bf.match(prev_des, curr_des)
                
                if len(matches) > 10:
                    # Compute feature positions
                    prev_pts = np.array([prev_kp[m.queryIdx].pt for m in matches])
                    curr_pts = np.array([curr_kp[m.trainIdx].pt for m in matches])
                    
                    # Compute displacement
                    displacements = np.sqrt(np.sum((curr_pts - prev_pts)**2, axis=1))
                    mean_disp = np.mean(displacements)
                    std_disp = np.std(displacements)
                    
                    # Stability: low variance in displacement = stable
                    stability = 1.0 / (1.0 + std_disp / (mean_disp + 1e-10))
                    stabilities.append(stability)
                    
                    # Drift: features that move inconsistently
                    # Check for non-rigid transformations
                    if len(prev_pts) > 4:
                        # Estimate affine transform
                        M, inliers = cv2.estimateAffinePartial2D(
                            prev_pts.astype(np.float32), 
                            curr_pts.astype(np.float32)
                        )
                        
                        if M is not None and inliers is not None:
                            inlier_ratio = np.sum(inliers) / len(inliers)
                            drift = 1.0 - inlier_ratio
                            drifts.append(drift)
            
            prev_gray = curr_gray
            prev_kp, prev_des = curr_kp, curr_des
        
        avg_stability = np.mean(stabilities) if stabilities else 1.0
        avg_drift = np.mean(drifts) if drifts else 0.0
        
        return {
            'stability': float(avg_stability),
            'drift': float(avg_drift)
        }
    
    def _analyze_motion_blur(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze motion blur consistency.
        Real cameras produce natural motion blur; AI may miss or incorrectly apply it.
        """
        if len(frames) < 2:
            return {'score': 1.0, 'anomalies': 0}
        
        anomalies = 0
        blur_scores = []
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Estimate blur using Laplacian variance
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Also check for directional blur using FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Check for directional patterns (motion blur creates streaks in FFT)
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Analyze radial vs tangential energy distribution
            y, x = np.ogrid[:h, :w]
            angle = np.arctan2(y - center_h, x - center_w)
            
            # Divide into angular sectors
            sector_energies = []
            for a in range(0, 360, 45):
                mask = (angle >= np.radians(a)) & (angle < np.radians(a + 45))
                sector_energies.append(np.sum(magnitude[mask]**2))
            
            # Motion blur creates imbalance in sector energies
            sector_energies = np.array(sector_energies)
            sector_var = np.var(sector_energies) / (np.mean(sector_energies) + 1e-10)
            
            blur_scores.append({
                'laplacian_var': lap_var,
                'directional_var': sector_var
            })
        
        # Overall score
        lap_vars = [s['laplacian_var'] for s in blur_scores]
        avg_lap = np.mean(lap_vars)
        score = min(avg_lap / 500, 1.0)  # Normalize
        
        # Sudden changes in blur level
        lap_changes = np.abs(np.diff(lap_vars))
        blur_change_threshold = np.mean(lap_vars) * 0.5
        anomalies = int(np.sum(lap_changes > blur_change_threshold))
        
        return {
            'score': float(score),
            'anomalies': int(anomalies)
        }
    
    def _analyze_temporal_noise(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze noise consistency across frames.
        AI videos may have inconsistent noise patterns.
        """
        if len(frames) < 2:
            return {'score': 1.0, 'inconsistency': 0.0}
        
        noise_stats = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # High-pass filter to get noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            noise = gray - blurred
            
            # Compute noise statistics
            noise_stats.append({
                'mean': np.mean(noise),
                'std': np.std(noise),
                'skew': self._compute_skewness(noise)
            })
        
        # Analyze consistency
        stds = [s['std'] for s in noise_stats]
        means = [s['mean'] for s in noise_stats]
        
        # Inconsistency = variance of noise statistics
        std_var = np.var(stds)
        mean_var = np.var(means)
        
        inconsistency = (std_var + mean_var) / (np.mean(stds)**2 + 1e-10)
        
        # Score: lower inconsistency = better
        score = 1.0 / (1.0 + inconsistency)
        
        return {
            'score': float(score),
            'inconsistency': float(inconsistency)
        }
    
    def _analyze_frame_transitions(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze frame-to-frame transitions for unnatural changes.
        """
        if len(frames) < 2:
            return {'score': 1.0, 'sudden_changes': 0}
        
        changes = []
        sudden_count = 0
        
        for i in range(1, len(frames)):
            prev = frames[i-1].astype(np.float64)
            curr = frames[i].astype(np.float64)
            
            # Compute frame difference
            diff = np.abs(curr - prev)
            change_ratio = np.mean(diff) / 255.0
            
            changes.append(change_ratio)
            
            # Detect sudden changes
            if i > 1:
                prev_change = changes[-2]
                if change_ratio > prev_change * 3 + 0.1:
                    sudden_count += 1
        
        # Score based on change smoothness
        change_std = np.std(changes)
        change_mean = np.mean(changes)
        
        # Smooth transitions have low std relative to mean
        score = 1.0 / (1.0 + change_std / (change_mean + 1e-10))
        
        return {
            'score': float(score),
            'sudden_changes': int(sudden_count)
        }
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std)**3))


def analyze_temporal(video_path: str) -> TemporalResult:
    """Convenience function for temporal analysis from video path"""
    analyzer = TemporalAnalyzer()
    return analyzer.analyze_video(video_path)


def analyze_temporal_frames(frames: List[np.ndarray]) -> TemporalResult:
    """Convenience function for temporal analysis from frame list"""
    analyzer = TemporalAnalyzer()
    return analyzer.analyze_frames(frames)
