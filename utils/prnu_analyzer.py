"""
PRNU (Photo Response Non-Uniformity) Analyzer for AI-Generated Content Detection

PRNU is a unique noise pattern caused by manufacturing variations in camera sensors.
Every camera leaves a consistent "fingerprint" in every photo it takes.

AI-generated images:
- Lack coherent sensor noise
- Show no device-consistent fingerprint
- Have synthetic noise patterns that don't match real sensor physics

Real photos:
- Have consistent PRNU patterns across images from same device
- Show physics-based noise characteristics
- Exhibit sensor-specific artifacts

This is one of the strongest forensic signals for detecting AI-generated content.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging
from scipy import ndimage
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PRNUResult:
    """Result of PRNU analysis"""
    is_suspicious: bool
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    prnu_present: bool = False
    noise_consistency: float = 0.0
    spectral_correlation: float = 0.0
    estimated_quality: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class PRNUAnalyzer:
    """
    Analyzes images for PRNU (sensor fingerprint) signatures.
    
    Key insight: Real camera images have consistent sensor noise patterns,
    while AI-generated images either lack these patterns or have synthetic
    noise that doesn't follow physics-based characteristics.
    """
    
    def __init__(self):
        # Thresholds based on research
        self.min_prnu_strength = 0.008  # Minimum PRNU correlation for real camera
        self.noise_consistency_threshold = 0.3
        self.spectral_threshold = 0.15
        
        # Wavelet denoising parameters
        self.wavelet = 'db8'  # Daubechies 8 wavelet
        self.denoise_level = 1
        
    def analyze_image(self, image: np.ndarray, 
                      reference_prnu: Optional[np.ndarray] = None) -> PRNUResult:
        """
        Analyze an image for PRNU characteristics.
        
        Args:
            image: RGB image as numpy array (H, W, C) or grayscale (H, W)
            reference_prnu: Optional known PRNU pattern to compare against
            
        Returns:
            PRNUResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = self._rgb_to_gray(image)
            else:
                gray = image.astype(np.float64)
            
            # Normalize
            gray = gray / 255.0 if gray.max() > 1 else gray
            
            # Step 1: Extract noise residual
            residual = self._extract_noise_residual(gray)
            details['residual_mean'] = float(np.abs(residual).mean())
            details['residual_std'] = float(residual.std())
            
            # Step 2: Estimate PRNU pattern
            estimated_prnu = self._estimate_prnu(residual, gray)
            
            # Step 3: Check noise characteristics
            noise_stats = self._analyze_noise_statistics(residual)
            details.update(noise_stats)
            
            # Step 4: Spectral analysis of noise
            spectral_score = self._analyze_noise_spectrum(residual)
            details['spectral_score'] = float(spectral_score)
            
            # Step 5: Check for PRNU consistency
            prnu_strength = self._compute_prnu_strength(estimated_prnu, residual)
            details['prnu_strength'] = float(prnu_strength)
            
            # Step 6: Cross-channel consistency (for color images)
            if len(image.shape) == 3:
                cross_channel_score = self._analyze_cross_channel_consistency(image)
                details['cross_channel_consistency'] = float(cross_channel_score)
            else:
                cross_channel_score = 1.0
                details['cross_channel_consistency'] = 1.0
            
            # Step 7: Compare with reference if provided
            if reference_prnu is not None:
                correlation = self._compute_prnu_correlation(estimated_prnu, reference_prnu)
                details['reference_correlation'] = float(correlation)
                
                if correlation > 0.4:
                    indicators.append(f"Strong match with reference PRNU ({correlation:.3f})")
                    prnu_present = True
                else:
                    indicators.append(f"No match with reference PRNU ({correlation:.3f})")
                    prnu_present = False
                    confidence += 0.3
            else:
                prnu_present = prnu_strength > self.min_prnu_strength
            
            # Analyze indicators
            
            # Check 1: PRNU strength
            if prnu_strength < self.min_prnu_strength:
                indicators.append(f"Weak/absent PRNU signal ({prnu_strength:.4f}) - suggests AI generation")
                confidence += 0.25
            else:
                indicators.append(f"PRNU signal present ({prnu_strength:.4f})")
            
            # Check 2: Noise consistency
            noise_consistency = noise_stats.get('noise_consistency', 0)
            if noise_consistency < self.noise_consistency_threshold:
                indicators.append(f"Inconsistent noise pattern ({noise_consistency:.3f})")
                confidence += 0.2
            
            # Check 3: Spectral characteristics
            if spectral_score < self.spectral_threshold:
                indicators.append(f"Unnatural noise spectrum ({spectral_score:.3f})")
                confidence += 0.2
            
            # Check 4: Cross-channel consistency
            if len(image.shape) == 3 and cross_channel_score < 0.5:
                indicators.append(f"Inconsistent noise across color channels ({cross_channel_score:.3f})")
                confidence += 0.15
            
            # Check 5: Noise variance patterns
            variance_ratio = noise_stats.get('variance_ratio', 1.0)
            if variance_ratio > 2.0 or variance_ratio < 0.5:
                indicators.append(f"Abnormal noise variance ratio ({variance_ratio:.2f})")
                confidence += 0.1
            
            # Check 6: High-frequency noise distribution
            hf_ratio = noise_stats.get('high_freq_ratio', 0)
            if hf_ratio > 0.7:
                indicators.append(f"Excessive high-frequency noise ({hf_ratio:.2f}) - possible GAN artifact")
                confidence += 0.15
            
            # Check 7: Blockiness detection
            blockiness = self._detect_blockiness(residual)
            details['blockiness'] = float(blockiness)
            if blockiness > 0.3:
                indicators.append("Inconsistent block artifacts - possible manipulation")
                confidence += 0.1
            
        except Exception as e:
            logger.error(f"PRNU analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return PRNUResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            prnu_present=prnu_present,
            noise_consistency=noise_stats.get('noise_consistency', 0),
            spectral_correlation=details.get('spectral_score', 0),
            estimated_quality=details.get('prnu_strength', 0),
            details=details
        )
    
    def _rgb_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale using standard weights"""
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1141])
    
    def _extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise residual using wavelet-based denoising.
        The residual = original - denoised contains sensor noise.
        """
        try:
            import pywt
            
            # Wavelet denoising
            coeffs = pywt.wavedec2(image, self.wavelet, level=self.denoise_level)
            
            # Threshold detail coefficients
            new_coeffs = [coeffs[0]]  # Keep approximation
            for i in range(1, len(coeffs)):
                # Soft thresholding
                detail = coeffs[i]
                sigma = np.median(np.abs(detail[0])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(image.size))
                
                new_detail = tuple(
                    pywt.threshold(d, threshold, mode='soft') 
                    for d in detail
                )
                new_coeffs.append(new_detail)
            
            # Reconstruct denoised image
            denoised = pywt.waverec2(new_coeffs, self.wavelet)
            
            # Ensure same size
            denoised = denoised[:image.shape[0], :image.shape[1]]
            
            # Residual is the noise
            residual = image - denoised
            
            return residual
            
        except ImportError:
            # Fallback: Use simple high-pass filter
            logger.warning("pywt not available, using simple high-pass filter")
            
            # Gaussian blur for low-pass
            blurred = ndimage.gaussian_filter(image, sigma=1.5)
            residual = image - blurred
            
            return residual
    
    def _estimate_prnu(self, residual: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Estimate PRNU pattern from noise residual.
        PRNU is multiplicative: noise = PRNU * signal
        So: PRNU ≈ residual / (original + epsilon)
        """
        epsilon = 1e-10
        prnu_estimate = residual / (original + epsilon)
        prnu_estimate = np.clip(prnu_estimate, -0.1, 0.1)
        return prnu_estimate
    
    def _analyze_noise_statistics(self, residual: np.ndarray) -> Dict[str, float]:
        """Analyze statistical properties of noise residual"""
        stats = {}
        
        # Basic statistics
        stats['noise_mean'] = float(np.mean(residual))
        stats['noise_std'] = float(np.std(residual))
        stats['noise_skewness'] = float(self._compute_skewness(residual))
        stats['noise_kurtosis'] = float(self._compute_kurtosis(residual))
        
        # Noise consistency across image regions
        h, w = residual.shape
        regions = [
            residual[:h//2, :w//2],
            residual[:h//2, w//2:],
            residual[h//2:, :w//2],
            residual[h//2:, w//2:]
        ]
        
        region_stds = [np.std(r) for r in regions]
        stats['noise_consistency'] = float(1.0 - (np.std(region_stds) / (np.mean(region_stds) + 1e-10)))
        
        # Variance ratio
        stats['variance_ratio'] = float(max(region_stds) / (min(region_stds) + 1e-10))
        
        # High-frequency content ratio
        fft = np.fft.fft2(residual)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        center_mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
        
        low_freq_energy = np.sum(magnitude[center_mask]**2)
        high_freq_energy = np.sum(magnitude[~center_mask]**2)
        
        stats['high_freq_ratio'] = float(high_freq_energy / (low_freq_energy + high_freq_energy + 1e-10))
        
        return stats
    
    def _analyze_noise_spectrum(self, residual: np.ndarray) -> float:
        """
        Analyze if noise spectrum matches expected sensor noise.
        Real sensor noise has specific spectral characteristics.
        """
        fft = np.fft.fft2(residual)
        power_spectrum = np.abs(fft)**2
        
        h, w = residual.shape
        center_h, center_w = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Bin by radius
        max_r = min(h, w) // 2
        radial_mean = np.zeros(max_r)
        
        for i in range(max_r):
            mask = (r >= i) & (r < i + 1)
            if np.any(mask):
                radial_mean[i] = np.mean(power_spectrum[mask])
        
        # Normalize
        radial_mean = radial_mean / (radial_mean[0] + 1e-10)
        
        # Expected: 1/f decay for natural noise
        freqs = np.arange(1, len(radial_mean))
        expected = 1.0 / (freqs + 1)
        
        correlation = np.corrcoef(radial_mean[1:], expected)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_prnu_strength(self, prnu_estimate: np.ndarray, residual: np.ndarray) -> float:
        """
        Compute the strength/quality of estimated PRNU.
        Strong PRNU indicates real camera, weak/absent indicates AI.
        """
        h, w = prnu_estimate.shape
        block_size = min(h, w) // 8
        
        correlations = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_prnu = prnu_estimate[i:i+block_size, j:j+block_size]
                block_residual = residual[i:i+block_size, j:j+block_size]
                
                corr = np.corrcoef(block_prnu.flatten(), block_residual.flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        strength = mean_corr * (1 - std_corr)
        return float(max(0, strength))
    
    def _analyze_cross_channel_consistency(self, image: np.ndarray) -> float:
        """
        Analyze noise consistency across color channels.
        Real cameras have correlated noise across RGB channels.
        """
        if image.shape[2] < 3:
            return 1.0
        
        residuals = []
        for c in range(3):
            channel = image[:, :, c].astype(np.float64) / 255.0
            residual = self._extract_noise_residual(channel)
            residuals.append(residual.flatten())
        
        correlations = []
        for i in range(3):
            for j in range(i + 1, 3):
                corr = np.corrcoef(residuals[i], residuals[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        return float(np.mean(correlations))
    
    def _compute_prnu_correlation(self, estimated: np.ndarray, reference: np.ndarray) -> float:
        """Compute correlation between estimated and reference PRNU"""
        if estimated.shape != reference.shape:
            reference = cv2.resize(reference, (estimated.shape[1], estimated.shape[0]))
        
        correlation = np.corrcoef(estimated.flatten(), reference.flatten())[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std)**3))
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std)**4) - 3)
    
    def _detect_blockiness(self, residual: np.ndarray) -> float:
        """
        Detect blockiness in noise residual.
        Inconsistent block artifacts suggest manipulation or AI generation.
        """
        h, w = residual.shape
        block_size = 8
        
        h_diffs = []
        v_diffs = []
        
        for i in range(block_size, h, block_size):
            diff = np.mean(np.abs(residual[i, :] - residual[i-1, :]))
            h_diffs.append(diff)
        
        for j in range(block_size, w, block_size):
            diff = np.mean(np.abs(residual[:, j] - residual[:, j-1]))
            v_diffs.append(diff)
        
        avg_diff = np.mean(np.abs(residual))
        
        if avg_diff < 1e-10:
            return 0.0
        
        h_blockiness = np.mean(h_diffs) / (avg_diff + 1e-10)
        v_blockiness = np.mean(v_diffs) / (avg_diff + 1e-10)
        
        return float((h_blockiness + v_blockiness) / 2)
    
    def compute_camera_fingerprint(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Compute PRNU fingerprint from multiple images known to be from same camera.
        More images = more accurate fingerprint.
        
        Args:
            images: List of images from same camera
            
        Returns:
            Estimated PRNU fingerprint
        """
        prnu_estimates = []
        
        for image in images:
            if len(image.shape) == 3:
                gray = self._rgb_to_gray(image)
            else:
                gray = image.astype(np.float64)
            
            gray = gray / 255.0 if gray.max() > 1 else gray
            
            residual = self._extract_noise_residual(gray)
            prnu = self._estimate_prnu(residual, gray)
            prnu_estimates.append(prnu)
        
        fingerprint = np.mean(prnu_estimates, axis=0)
        return fingerprint


def analyze_prnu(image: np.ndarray, reference_prnu: Optional[np.ndarray] = None) -> PRNUResult:
    """Convenience function for PRNU analysis"""
    analyzer = PRNUAnalyzer()
    return analyzer.analyze_image(image, reference_prnu)
