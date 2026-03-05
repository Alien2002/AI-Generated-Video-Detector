"""
Frequency Domain Analyzer for AI-Generated Content Detection

Analyzes frequency patterns that are often distinct in AI-generated content:
- GAN-generated images often have regular frequency artifacts
- Diffusion models may have specific spectral signatures
- Real photos have natural frequency distributions
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrequencyResult:
    """Result of frequency analysis"""
    is_suspicious: bool
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    spectral_features: Dict[str, float] = field(default_factory=dict)


class FrequencyAnalyzer:
    """Analyzes frequency domain patterns for AI detection"""
    
    def __init__(self):
        # Thresholds based on research (adjust based on your data)
        self.high_freq_threshold = 0.15  # AI images often have less high-freq content
        self.regularity_threshold = 0.25  # GAN images have more regular patterns
        self.spectral_flatness_threshold = 0.4
    
    def analyze_image(self, image: np.ndarray) -> FrequencyResult:
        """
        Analyze image in frequency domain.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            
        Returns:
            FrequencyResult with analysis details
        """
        indicators = []
        spectral_features = {}
        confidence = 0.0
        
        try:
            # Convert to grayscale if color
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Compute 2D FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Normalize
            magnitude_spectrum = magnitude_spectrum / (magnitude_spectrum.max() + 1e-10)
            
            # Feature 1: High frequency content ratio
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            radius = min(h, w) // 4
            
            # Create masks
            y, x = np.ogrid[:h, :w]
            center_mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
            high_freq_mask = ~center_mask
            
            low_freq_energy = np.sum(magnitude_spectrum[center_mask]**2)
            high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]**2)
            total_energy = low_freq_energy + high_freq_energy + 1e-10
            
            high_freq_ratio = high_freq_energy / total_energy
            spectral_features['high_freq_ratio'] = float(high_freq_ratio)
            
            # AI images often have lower high-frequency content
            if high_freq_ratio < self.high_freq_threshold:
                indicators.append(f"Low high-frequency content ({high_freq_ratio:.3f})")
                confidence += 0.2
            
            # Feature 2: Spectral regularity (GANs produce regular patterns)
            regularity = self._compute_spectral_regularity(magnitude_spectrum)
            spectral_features['spectral_regularity'] = float(regularity)
            
            if regularity > self.regularity_threshold:
                indicators.append(f"High spectral regularity ({regularity:.3f}) - possible GAN")
                confidence += 0.25
            
            # Feature 3: Spectral flatness measure
            flatness = self._compute_spectral_flatness(magnitude_spectrum)
            spectral_features['spectral_flatness'] = float(flatness)
            
            if flatness > self.spectral_flatness_threshold:
                indicators.append(f"Unusual spectral flatness ({flatness:.3f})")
                confidence += 0.15
            
            # Feature 4: Check for grid-like artifacts (common in GANs)
            grid_score = self._detect_grid_artifacts(magnitude_spectrum)
            spectral_features['grid_artifact_score'] = float(grid_score)
            
            if grid_score > 0.3:
                indicators.append(f"Grid-like artifacts detected ({grid_score:.3f})")
                confidence += 0.2
            
            # Feature 5: Analyze frequency band distribution
            band_features = self._analyze_frequency_bands(magnitude_spectrum)
            spectral_features.update(band_features)
            
            # Check for unusual band distributions
            if band_features.get('band_ratio_anomaly', False):
                indicators.append("Unusual frequency band distribution")
                confidence += 0.15
            
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return FrequencyResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            spectral_features=spectral_features
        )
    
    def _compute_spectral_regularity(self, magnitude: np.ndarray) -> float:
        """Compute spectral regularity score (higher = more regular = more likely GAN)"""
        # Look for repeating patterns in the spectrum
        h, w = magnitude.shape
        
        # Sample horizontal and vertical slices
        h_slice = magnitude[h//2, :]
        v_slice = magnitude[:, w//2]
        
        # Compute autocorrelation to detect regularity
        h_autocorr = np.correlate(h_slice, h_slice, mode='full')
        v_autocorr = np.correlate(v_slice, v_slice, mode='full')
        
        # Normalize
        h_autocorr = h_autocorr / (h_autocorr.max() + 1e-10)
        v_autocorr = v_autocorr / (v_autocorr.max() + 1e-10)
        
        # Look for secondary peaks (indicates regularity)
        h_peaks = self._count_significant_peaks(h_autocorr)
        v_peaks = self._count_significant_peaks(v_autocorr)
        
        regularity = (h_peaks + v_peaks) / 4.0  # Normalize to 0-1
        return min(regularity, 1.0)
    
    def _count_significant_peaks(self, signal: np.ndarray, threshold: float = 0.3) -> int:
        """Count peaks above threshold in normalized signal"""
        # Skip center peak
        mid = len(signal) // 2
        signal = np.delete(signal, mid)
        
        # Count peaks
        peaks = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > threshold:
                    peaks += 1
        return peaks
    
    def _compute_spectral_flatness(self, magnitude: np.ndarray) -> float:
        """Compute spectral flatness (geometric mean / arithmetic mean)"""
        flat_magnitude = magnitude.flatten() + 1e-10
        
        geometric_mean = np.exp(np.mean(np.log(flat_magnitude)))
        arithmetic_mean = np.mean(flat_magnitude)
        
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return float(flatness)
    
    def _detect_grid_artifacts(self, magnitude: np.ndarray) -> float:
        """Detect grid-like artifacts common in GAN-generated images"""
        h, w = magnitude.shape
        
        # Look for periodic peaks in frequency domain
        # These appear as bright spots in a grid pattern
        
        # Threshold to find bright spots
        threshold = magnitude.mean() + magnitude.std()
        bright_spots = (magnitude > threshold).astype(float)
        
        # Sum along axes to detect grid patterns
        h_profile = bright_spots.sum(axis=0)
        v_profile = bright_spots.sum(axis=1)
        
        # Look for periodic patterns
        h_peaks = self._count_significant_peaks(h_profile / (h_profile.max() + 1e-10))
        v_peaks = self._count_significant_peaks(v_profile / (v_profile.max() + 1e-10))
        
        # More peaks = more grid-like = more likely GAN
        grid_score = (h_peaks + v_peaks) / 8.0
        return min(grid_score, 1.0)
    
    def _analyze_frequency_bands(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze energy distribution across frequency bands"""
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Define frequency bands
        bands = {}
        radii = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # As fraction of image size
        min_dim = min(h, w)
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        prev_radius = 0
        band_energies = []
        
        for r in radii:
            radius_pixels = int(r * min_dim / 2)
            mask = (dist_from_center > prev_radius) & (dist_from_center <= radius_pixels)
            energy = np.sum(magnitude[mask]**2)
            band_energies.append(energy)
            bands[f'band_{int(r*100)}'] = float(energy)
            prev_radius = radius_pixels
        
        # Normalize band energies
        total = sum(band_energies) + 1e-10
        band_ratios = [e / total for e in band_energies]
        
        # Check for anomalies (AI images often have unusual distribution)
        # Real photos typically have more energy in lower bands
        # AI images may have more uniform distribution
        
        low_band_ratio = band_ratios[0] + band_ratios[1]  # First two bands
        high_band_ratio = band_ratios[-2] + band_ratios[-1]  # Last two bands
        
        bands['low_band_ratio'] = float(low_band_ratio)
        bands['high_band_ratio'] = float(high_band_ratio)
        
        # Anomaly if high bands have too much energy
        bands['band_ratio_anomaly'] = high_band_ratio > 0.3
        
        return bands


def analyze_frequency(image: np.ndarray) -> FrequencyResult:
    """Convenience function for frequency analysis"""
    analyzer = FrequencyAnalyzer()
    return analyzer.analyze_image(image)
