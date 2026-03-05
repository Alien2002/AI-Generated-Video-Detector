"""
CFA (Color Filter Array) / Demosaicing Analyzer for AI-Generated Content Detection

Real cameras use a Bayer color filter array (CFA) to capture color images.
The raw sensor data has only one color channel per pixel (R, G, or B).
Demosaicing algorithms interpolate the missing colors, leaving characteristic artifacts.

AI-generated images:
- Never pass through a real CFA
- Lack Bayer interpolation artifacts
- Show no periodic color correlation patterns
- Have "perfect" pixel-level color that doesn't match physics

Real photos:
- Exhibit CFA interpolation traces
- Have periodic color correlation patterns
- Show demosaicing artifacts in high-frequency regions
- Follow physics-based color sampling constraints

This is a strong physics-based forensic signal.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class CFAResult:
    """Result of CFA/Demosaicing analysis"""
    is_suspicious: bool
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    cfa_detected: bool = False
    bayer_pattern: str = "unknown"
    correlation_score: float = 0.0
    interpolation_artifacts: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class CFAAnalyzer:
    """
    Analyzes images for Color Filter Array (CFA) / demosaicing artifacts.
    
    Key insight: Real camera images have characteristic interpolation patterns
    from the Bayer filter, while AI-generated images lack these physics-based traces.
    """
    
    def __init__(self):
        # Standard Bayer patterns (RGGB is most common)
        self.bayer_patterns = {
            'RGGB': np.array([[0, 1], [1, 2]]),  # 0=R, 1=G, 2=B
            'GRBG': np.array([[1, 0], [2, 1]]),
            'GBRG': np.array([[1, 2], [0, 1]]),
            'BGGR': np.array([[2, 1], [1, 0]]),
        }
        
        # Thresholds
        self.min_correlation = 0.15  # Minimum for CFA detection
        self.min_artifacts = 0.08   # Minimum interpolation artifacts
        
    def analyze_image(self, image: np.ndarray) -> CFAResult:
        """
        Analyze an image for CFA/demosaicing characteristics.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            
        Returns:
            CFAResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        if len(image.shape) != 3 or image.shape[2] < 3:
            return CFAResult(
                is_suspicious=False,
                confidence_score=0.0,
                indicators=["Image is not RGB - CFA analysis not applicable"],
                cfa_detected=False
            )
        
        try:
            # Normalize to 0-1
            img = image.astype(np.float64)
            if img.max() > 1:
                img = img / 255.0
            
            # Extract color channels
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            
            # Step 1: Check for periodic color correlations (CFA signature)
            correlation_scores = self._check_color_correlations(r, g, b)
            details['color_correlations'] = correlation_scores
            
            max_correlation = max(correlation_scores.values())
            details['max_correlation'] = float(max_correlation)
            
            # Step 2: Detect interpolation artifacts
            interp_score = self._detect_interpolation_artifacts(r, g, b)
            details['interpolation_score'] = float(interp_score)
            
            # Step 3: Analyze high-frequency color differences
            hf_color_diff = self._analyze_hf_color_differences(r, g, b)
            details['hf_color_diff'] = hf_color_diff
            
            # Step 4: Check for CFA pattern periodicity
            pattern_scores = self._detect_bayer_pattern(r, g, b)
            details['pattern_scores'] = pattern_scores
            
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            best_score = pattern_scores[best_pattern]
            details['best_pattern'] = best_pattern
            details['pattern_score'] = float(best_score)
            
            # Step 5: Analyze local color variance (demosaicing leaves traces)
            variance_score = self._analyze_local_color_variance(img)
            details['local_variance_score'] = float(variance_score)
            
            # Step 6: Check chroma subsampling effects
            chroma_score = self._check_chroma_patterns(r, g, b)
            details['chroma_pattern_score'] = float(chroma_score)
            
            # Determine if CFA is present
            cfa_detected = (max_correlation > self.min_correlation or 
                          interp_score > self.min_artifacts or
                          best_score > 0.2)
            
            # Generate indicators
            
            # Check 1: Color correlations
            if max_correlation < self.min_correlation:
                indicators.append(f"Low periodic color correlation ({max_correlation:.3f}) - no CFA pattern")
                confidence += 0.2
            else:
                indicators.append(f"Periodic color correlation detected ({max_correlation:.3f})")
            
            # Check 2: Interpolation artifacts
            if interp_score < self.min_artifacts:
                indicators.append(f"Missing demosaicing artifacts ({interp_score:.3f}) - suggests AI")
                confidence += 0.25
            else:
                indicators.append(f"Demosaicing artifacts present ({interp_score:.3f})")
            
            # Check 3: High-frequency color differences
            if hf_color_diff.get('anomaly', False):
                indicators.append("Abnormal high-frequency color relationships")
                confidence += 0.15
            
            # Check 4: Bayer pattern detection
            if best_score < 0.15:
                indicators.append(f"No detectable Bayer pattern ({best_score:.3f})")
                confidence += 0.2
            else:
                indicators.append(f"Bayer pattern detected: {best_pattern} ({best_score:.3f})")
            
            # Check 5: Local variance
            if variance_score < 0.1:
                indicators.append("Unusually uniform local color variance - possible AI generation")
                confidence += 0.1
            
            # Check 6: Chroma patterns
            if chroma_score < 0.1:
                indicators.append("Missing chroma subsampling patterns")
                confidence += 0.1
            
        except Exception as e:
            logger.error(f"CFA analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return CFAResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            cfa_detected=cfa_detected,
            bayer_pattern=details.get('best_pattern', 'unknown'),
            correlation_score=details.get('max_correlation', 0),
            interpolation_artifacts=details.get('interpolation_score', 0),
            details=details
        )
    
    def _check_color_correlations(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        """
        Check for periodic correlations between color channels.
        CFA creates characteristic correlations at specific pixel offsets.
        """
        correlations = {}
        
        # Downsample to reduce noise
        r_ds = r[::2, ::2]
        g_ds = g[::2, ::2]
        b_ds = b[::2, ::2]
        
        # Check correlations at offset (1,0), (0,1), (1,1)
        # These are expected from Bayer pattern
        
        # R-G correlation
        r_g_shifted = np.corrcoef(r_ds.flatten(), np.roll(g_ds, 1, axis=0).flatten())[0, 1]
        correlations['r_g_vertical'] = float(abs(r_g_shifted)) if not np.isnan(r_g_shifted) else 0
        
        # G-B correlation
        g_b_shifted = np.corrcoef(g_ds.flatten(), np.roll(b_ds, 1, axis=1).flatten())[0, 1]
        correlations['g_b_horizontal'] = float(abs(g_b_shifted)) if not np.isnan(g_b_shifted) else 0
        
        # R-B diagonal correlation
        r_b_diag = np.corrcoef(r_ds.flatten(), np.roll(np.roll(b_ds, 1, axis=0), 1, axis=1).flatten())[0, 1]
        correlations['r_b_diagonal'] = float(abs(r_b_diag)) if not np.isnan(r_b_diag) else 0
        
        # G-G correlation (green pixels are at (0,1) and (1,0) in RGGB)
        g1 = g[::2, 1::2]
        g2 = g[1::2, ::2]
        g_corr = np.corrcoef(g1.flatten()[:min(g1.size, 10000)], g2.flatten()[:min(g2.size, 10000)])[0, 1]
        correlations['g_g_correlation'] = float(abs(g_corr)) if not np.isnan(g_corr) else 0
        
        return correlations
    
    def _detect_interpolation_artifacts(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
        """
        Detect artifacts from demosaicing interpolation.
        These appear as periodic patterns in high-frequency content.
        """
        # Compute high-frequency content for each channel
        hf_r = self._high_pass_filter(r)
        hf_g = self._high_pass_filter(g)
        hf_b = self._high_pass_filter(b)
        
        # Look for periodic patterns in high-frequency content
        # These should have 2x2 periodicity from Bayer pattern
        
        # Compute autocorrelation at lag 2
        def autocorr_lag2(signal):
            h, w = signal.shape
            # Horizontal lag-2
            h_corr = np.corrcoef(signal[:, :-2].flatten(), signal[:, 2:].flatten())[0, 1]
            # Vertical lag-2
            v_corr = np.corrcoef(signal[:-2, :].flatten(), signal[2:, :].flatten())[0, 1]
            return (abs(h_corr) + abs(v_corr)) / 2 if not (np.isnan(h_corr) or np.isnan(v_corr)) else 0
        
        r_periodicity = autocorr_lag2(hf_r)
        g_periodicity = autocorr_lag2(hf_g)
        b_periodicity = autocorr_lag2(hf_b)
        
        # Average periodicity score
        avg_periodicity = (r_periodicity + g_periodicity + b_periodicity) / 3
        
        return float(avg_periodicity)
    
    def _high_pass_filter(self, img: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to extract high-frequency content"""
        # Gaussian blur for low-pass
        low_pass = ndimage.gaussian_filter(img, sigma=1.0)
        # High-pass = original - low-pass
        high_pass = img - low_pass
        return high_pass
    
    def _analyze_hf_color_differences(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Analyze high-frequency color differences.
        Real cameras have correlated HF content across channels due to CFA.
        """
        hf_r = self._high_pass_filter(r)
        hf_g = self._high_pass_filter(g)
        hf_b = self._high_pass_filter(b)
        
        # Compute ratios
        rg_ratio = np.abs(hf_r) / (np.abs(hf_g) + 1e-10)
        gb_ratio = np.abs(hf_g) / (np.abs(hf_b) + 1e-10)
        
        # Check for anomalies
        rg_std = np.std(rg_ratio)
        gb_std = np.std(gb_ratio)
        
        # Real photos have more consistent ratios
        anomaly = (rg_std > 2.0 or gb_std > 2.0)
        
        return {
            'rg_ratio_std': float(rg_std),
            'gb_ratio_std': float(gb_std),
            'anomaly': anomaly
        }
    
    def _detect_bayer_pattern(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        """
        Try to detect which Bayer pattern was used.
        Each pattern creates different correlations.
        """
        scores = {}
        
        h, w = r.shape
        
        for pattern_name, pattern in self.bayer_patterns.items():
            # Check if color values match expected pattern
            # At each position, one channel should have "native" values
            # while others are interpolated
            
            score = 0.0
            
            # Sample positions
            for i in range(0, min(h-2, 256), 2):
                for j in range(0, min(w-2, 256), 2):
                    # Check variance at each position
                    for pi in range(2):
                        for pj in range(2):
                            channel = pattern[pi, pj]
                            
                            # Native pixels should have higher local variance
                            # (interpolated pixels are smoother)
                            if channel == 0:  # R
                                native_var = self._local_variance(r, i+pi, j+pj)
                                interp_var_g = self._local_variance(g, i+pi, j+pj)
                                interp_var_b = self._local_variance(b, i+pi, j+pj)
                            elif channel == 1:  # G
                                native_var = self._local_variance(g, i+pi, j+pj)
                                interp_var_r = self._local_variance(r, i+pi, j+pj)
                                interp_var_b = self._local_variance(b, i+pi, j+pj)
                            else:  # B
                                native_var = self._local_variance(b, i+pi, j+pj)
                                interp_var_r = self._local_variance(r, i+pi, j+pj)
                                interp_var_g = self._local_variance(g, i+pi, j+pj)
                            
                            # Native should have higher variance than interpolated
                            if channel == 0:
                                score += (native_var > interp_var_g) + (native_var > interp_var_b)
                            elif channel == 1:
                                score += (native_var > interp_var_r) + (native_var > interp_var_b)
                            else:
                                score += (native_var > interp_var_r) + (native_var > interp_var_g)
            
            # Normalize score
            max_score = (min(h-2, 256) // 2) * (min(w-2, 256) // 2) * 4 * 2
            scores[pattern_name] = float(score / max_score) if max_score > 0 else 0
        
        return scores
    
    def _local_variance(self, img: np.ndarray, i: int, j: int, size: int = 3) -> float:
        """Compute local variance around a pixel"""
        h, w = img.shape
        i_start = max(0, i - size // 2)
        i_end = min(h, i + size // 2 + 1)
        j_start = max(0, j - size // 2)
        j_end = min(w, j + size // 2 + 1)
        
        local_region = img[i_start:i_end, j_start:j_end]
        return float(np.var(local_region))
    
    def _analyze_local_color_variance(self, img: np.ndarray) -> float:
        """
        Analyze local variance patterns across color channels.
        Demosaicing creates characteristic variance patterns.
        """
        # Compute local variance for each channel
        kernel_size = 5
        
        def local_var(channel):
            # Use uniform filter for mean
            mean = ndimage.uniform_filter(channel, size=kernel_size)
            # Variance = E[X^2] - E[X]^2
            mean_sq = ndimage.uniform_filter(channel**2, size=kernel_size)
            var = mean_sq - mean**2
            return var
        
        var_r = local_var(img[:, :, 0])
        var_g = local_var(img[:, :, 1])
        var_b = local_var(img[:, :, 2])
        
        # Check for periodic variance patterns (2x2)
        # Green channel should have higher variance at more positions
        g_higher_r = np.mean(var_g > var_r)
        g_higher_b = np.mean(var_g > var_b)
        
        # In RGGB pattern, G has 2 positions per 2x2 block
        # So G should be higher ~50% of the time
        # Too uniform suggests AI
        g_ratio = (g_higher_r + g_higher_b) / 2
        
        # Score based on how close to expected 0.5
        score = 1.0 - abs(g_ratio - 0.5) * 2
        
        return float(score)
    
    def _check_chroma_patterns(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
        """
        Check for chroma subsampling patterns.
        Many cameras use 4:2:0 or 4:2:2 subsampling which leaves traces.
        """
        # Chroma is typically subsampled, so R and B should have
        # lower high-frequency content than G
        
        hf_r = self._high_pass_filter(r)
        hf_g = self._high_pass_filter(g)
        hf_b = self._high_pass_filter(b)
        
        hf_r_energy = np.mean(hf_r**2)
        hf_g_energy = np.mean(hf_g**2)
        hf_b_energy = np.mean(hf_b**2)
        
        # G should have highest HF energy
        # R and B should be similar to each other
        
        total_energy = hf_r_energy + hf_g_energy + hf_b_energy + 1e-10
        g_ratio = hf_g_energy / total_energy
        
        # G should be ~40-50% of HF energy in real photos
        # AI images often have more uniform distribution
        
        score = 1.0 - abs(g_ratio - 0.45) * 2
        
        return float(max(0, score))


def analyze_cfa(image: np.ndarray) -> CFAResult:
    """Convenience function for CFA analysis"""
    analyzer = CFAAnalyzer()
    return analyzer.analyze_image(image)
