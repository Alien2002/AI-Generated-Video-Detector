"""
Noise Residual Analyzer for AI-Generated Content Detection

Analyzes noise patterns in images, video frames, and audio to detect AI generation.

Key insight: AI-generated content has different noise characteristics than natural content:
- Images: Synthetic noise patterns, uniform variance, lack of natural sensor noise
- Video: Inconsistent noise across frames, temporal noise anomalies
- Audio: Over-regular noise floor, missing natural microstructure

This analyzer extracts high-frequency residuals and analyzes their statistical properties.
"""

import numpy as np
from typing import Dict, List, Any, Union
from dataclasses import dataclass, field
import logging
from scipy import ndimage
from scipy.signal import welch

logger = logging.getLogger(__name__)


@dataclass
class NoiseResult:
    """Result of noise residual analysis"""
    is_suspicious: bool
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    noise_type: str = "unknown"  # 'image', 'video', 'audio'
    noise_uniformity: float = 0.0
    noise_naturalness: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class NoiseAnalyzer:
    """
    Analyzes noise residuals for AI-generated content detection.
    Works on images, video frames, and audio signals.
    """
    
    def __init__(self):
        # Thresholds
        self.uniformity_threshold = 0.7  # High uniformity = suspicious
        self.naturalness_threshold = 0.3  # Low naturalness = suspicious
        
    def analyze(self, media: np.ndarray, media_type: str = "auto") -> NoiseResult:
        """
        Analyze noise in any media type.
        
        Args:
            media: Input media (image array, video frame list, or audio signal)
            media_type: 'image', 'video', 'audio', or 'auto' for auto-detection
            
        Returns:
            NoiseResult with analysis details
        """
        # Auto-detect media type
        if media_type == "auto":
            media_type = self._detect_media_type(media)
        
        if media_type == "image":
            return self.analyze_image(media)
        elif media_type == "video":
            return self.analyze_video(media)
        elif media_type == "audio":
            return self.analyze_audio(media)
        else:
            return NoiseResult(
                is_suspicious=False,
                confidence_score=0.0,
                indicators=[f"Unknown media type: {media_type}"],
                noise_type=media_type
            )
    
    def _detect_media_type(self, media: np.ndarray) -> str:
        """Auto-detect media type from array shape"""
        if media is None:
            return "unknown"
        
        shape = media.shape if hasattr(media, 'shape') else ()
        
        if len(shape) == 3:
            if shape[2] in [1, 3, 4]:
                return "image"
            else:
                return "video"  # Could be video frames
        elif len(shape) == 2:
            # Could be grayscale image or audio spectrogram
            if shape[0] > 100 and shape[1] > 100:
                return "image"
            else:
                return "audio"
        elif len(shape) == 1:
            return "audio"
        else:
            return "unknown"
    
    def analyze_image(self, image: np.ndarray) -> NoiseResult:
        """
        Analyze noise in an image.
        
        Args:
            image: RGB or grayscale image as numpy array
            
        Returns:
            NoiseResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        try:
            # Normalize
            img = image.astype(np.float64)
            if img.max() > 1:
                img = img / 255.0
            
            # Extract noise residual
            if len(img.shape) == 3:
                # Process each channel
                residuals = []
                for c in range(min(img.shape[2], 3)):
                    residuals.append(self._extract_noise_residual_2d(img[:, :, c]))
                residual = np.stack(residuals, axis=-1)
            else:
                residual = self._extract_noise_residual_2d(img)
            
            details['residual_mean'] = float(np.abs(residual).mean())
            details['residual_std'] = float(residual.std())
            
            # Analyze noise properties
            uniformity = self._analyze_noise_uniformity(residual)
            details['noise_uniformity'] = float(uniformity)
            
            naturalness = self._analyze_noise_naturalness(residual)
            details['noise_naturalness'] = float(naturalness)
            
            # Check for synthetic noise patterns
            synthetic_score = self._detect_synthetic_patterns(residual)
            details['synthetic_score'] = float(synthetic_score)
            
            # Analyze noise variance distribution
            variance_analysis = self._analyze_variance_distribution(residual)
            details.update(variance_analysis)
            
            # Check for banding/posterization
            banding_score = self._detect_banding(image)
            details['banding_score'] = float(banding_score)
            
            # Generate indicators
            
            # Check 1: Noise uniformity
            if uniformity > self.uniformity_threshold:
                indicators.append(f"Highly uniform noise ({uniformity:.3f}) - suggests AI generation")
                confidence += 0.25
            else:
                indicators.append(f"Natural noise variation ({uniformity:.3f})")
            
            # Check 2: Noise naturalness
            if naturalness < self.naturalness_threshold:
                indicators.append(f"Unnatural noise characteristics ({naturalness:.3f})")
                confidence += 0.2
            
            # Check 3: Synthetic patterns
            if synthetic_score > 0.5:
                indicators.append(f"Synthetic noise patterns detected ({synthetic_score:.3f})")
                confidence += 0.25
            
            # Check 4: Variance distribution
            if variance_analysis.get('variance_anomaly', False):
                indicators.append("Abnormal noise variance distribution")
                confidence += 0.15
            
            # Check 5: Banding
            if banding_score > 0.3:
                indicators.append(f"Banding/posterization detected ({banding_score:.3f}) - possible AI artifact")
                confidence += 0.15
            
        except Exception as e:
            logger.error(f"Image noise analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return NoiseResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            noise_type="image",
            noise_uniformity=details.get('noise_uniformity', 0),
            noise_naturalness=details.get('noise_naturalness', 0),
            details=details
        )
    
    def analyze_video(self, video_frames: Union[List[np.ndarray], np.ndarray]) -> NoiseResult:
        """
        Analyze noise across video frames.
        
        Args:
            video_frames: List of frames or array of shape (N, H, W, C)
            
        Returns:
            NoiseResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        try:
            # Convert to list if numpy array
            if isinstance(video_frames, np.ndarray):
                if len(video_frames.shape) == 4:
                    frames = [video_frames[i] for i in range(video_frames.shape[0])]
                else:
                    frames = [video_frames]
            else:
                frames = video_frames
            
            if len(frames) == 0:
                return NoiseResult(is_suspicious=False, confidence_score=0.0, 
                                 indicators=["No frames provided"], noise_type="video")
            
            # Analyze noise in each frame
            frame_results = []
            for i, frame in enumerate(frames[:30]):  # Limit to 30 frames
                result = self.analyze_image(frame)
                frame_results.append({
                    'confidence': result.confidence_score,
                    'uniformity': result.noise_uniformity,
                    'naturalness': result.noise_naturalness
                })
            
            # Analyze temporal consistency
            uniformities = [r['uniformity'] for r in frame_results]
            naturalnesses = [r['naturalness'] for r in frame_results]
            
            # Temporal variance of noise properties
            uniformity_temporal_var = np.var(uniformities)
            naturalness_temporal_var = np.var(naturalnesses)
            
            details['uniformity_temporal_variance'] = float(uniformity_temporal_var)
            details['naturalness_temporal_variance'] = float(naturalness_temporal_var)
            details['mean_frame_confidence'] = float(np.mean([r['confidence'] for r in frame_results]))
            
            # Check for temporal anomalies
            # Real video: noise properties should be relatively stable
            # AI video: may have inconsistent noise across frames
            
            if uniformity_temporal_var > 0.01:
                indicators.append(f"Inconsistent noise uniformity across frames ({uniformity_temporal_var:.4f})")
                confidence += 0.2
            
            if naturalness_temporal_var > 0.02:
                indicators.append(f"Varying noise naturalness across frames ({naturalness_temporal_var:.4f})")
                confidence += 0.2
            
            # Average frame-level detection
            avg_confidence = details['mean_frame_confidence']
            if avg_confidence > 0.3:
                indicators.append(f"Frame-level analysis suggests AI content ({avg_confidence:.2f})")
                confidence += avg_confidence * 0.5
            
            # Check for frame-to-frame noise correlation
            if len(frames) >= 2:
                noise_correlation = self._compute_frame_noise_correlation(frames[:10])
                details['frame_noise_correlation'] = float(noise_correlation)
                
                if noise_correlation > 0.8:
                    indicators.append(f"Unusually high frame-to-frame noise correlation ({noise_correlation:.2f})")
                    confidence += 0.15
            
        except Exception as e:
            logger.error(f"Video noise analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return NoiseResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            noise_type="video",
            noise_uniformity=details.get('mean_frame_confidence', 0),
            noise_naturalness=0,
            details=details
        )
    
    def analyze_audio(self, audio: np.ndarray, sample_rate: int = 22050) -> NoiseResult:
        """
        Analyze noise in audio signal.
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            NoiseResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        try:
            # Ensure 1D
            audio = audio.flatten().astype(np.float64)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
            # Extract noise floor
            noise_floor = self._extract_audio_noise_floor(audio)
            details['noise_floor_level'] = float(noise_floor)
            
            # Analyze noise floor consistency
            noise_uniformity = self._analyze_audio_noise_uniformity(audio)
            details['audio_noise_uniformity'] = float(noise_uniformity)
            
            # Analyze spectral properties of noise
            spectral_props = self._analyze_audio_noise_spectrum(audio, sample_rate)
            details.update(spectral_props)
            
            # Detect synthetic audio characteristics
            synthetic_score = self._detect_synthetic_audio(audio, sample_rate)
            details['audio_synthetic_score'] = float(synthetic_score)
            
            # Analyze microstructure (breath, pauses, etc.)
            microstructure = self._analyze_audio_microstructure(audio)
            details['microstructure_score'] = float(microstructure)
            
            # Generate indicators
            
            # Check 1: Noise floor
            if noise_floor < -60:  # Very low noise floor
                indicators.append(f"Suspiciously low noise floor ({noise_floor:.1f} dB) - possible AI")
                confidence += 0.2
            elif noise_floor > -20:  # Very high noise floor
                indicators.append(f"High noise floor ({noise_floor:.1f} dB)")
            
            # Check 2: Noise uniformity
            if noise_uniformity > 0.8:
                indicators.append(f"Over-uniform noise pattern ({noise_uniformity:.3f}) - AI speech characteristic")
                confidence += 0.25
            
            # Check 3: Spectral anomalies
            if spectral_props.get('spectral_anomaly', False):
                indicators.append("Abnormal noise spectral distribution")
                confidence += 0.15
            
            # Check 4: Synthetic characteristics
            if synthetic_score > 0.5:
                indicators.append(f"Synthetic audio characteristics detected ({synthetic_score:.3f})")
                confidence += 0.25
            
            # Check 5: Microstructure
            if microstructure < 0.3:
                indicators.append(f"Missing natural audio microstructure ({microstructure:.3f})")
                confidence += 0.2
            
        except Exception as e:
            logger.error(f"Audio noise analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return NoiseResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            noise_type="audio",
            noise_uniformity=details.get('audio_noise_uniformity', 0),
            noise_naturalness=details.get('microstructure_score', 0),
            details=details
        )
    
    def _extract_noise_residual_2d(self, image: np.ndarray) -> np.ndarray:
        """Extract noise residual using high-pass filtering"""
        # Gaussian blur for low-pass
        low_pass = ndimage.gaussian_filter(image, sigma=1.5)
        # High-pass = original - low-pass
        residual = image - low_pass
        return residual
    
    def _analyze_noise_uniformity(self, residual: np.ndarray) -> float:
        """
        Analyze how uniform the noise is across the image.
        AI-generated images tend to have more uniform noise.
        """
        # Divide into blocks and compute local noise variance
        if len(residual.shape) == 3:
            residual = np.mean(residual, axis=2)
        
        h, w = residual.shape
        block_size = min(h, w) // 8
        
        variances = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = residual[i:i+block_size, j:j+block_size]
                variances.append(np.var(block))
        
        if not variances:
            return 0.0
        
        # Uniformity = low variance of variances
        var_of_vars = np.var(variances)
        mean_var = np.mean(variances)
        
        # Normalize: lower variance ratio = more uniform
        uniformity = 1.0 / (1.0 + var_of_vars / (mean_var + 1e-10))
        
        return float(uniformity)
    
    def _analyze_noise_naturalness(self, residual: np.ndarray) -> float:
        """
        Analyze if noise follows natural distribution.
        Natural sensor noise follows specific statistical properties.
        """
        # Flatten residual
        flat = residual.flatten()
        
        # Check if distribution is Gaussian-like (natural noise)
        # Using kurtosis: Gaussian has kurtosis = 0
        mean = np.mean(flat)
        std = np.std(flat)
        
        if std < 1e-10:
            return 0.0
        
        # Compute kurtosis
        kurtosis = np.mean(((flat - mean) / std)**4) - 3
        
        # Compute skewness
        skewness = np.mean(((flat - mean) / std)**3)
        
        # Natural noise: kurtosis near 0, skewness near 0
        naturalness = 1.0 / (1.0 + abs(kurtosis) / 3 + abs(skewness))
        
        return float(naturalness)
    
    def _detect_synthetic_patterns(self, residual: np.ndarray) -> float:
        """
        Detect synthetic noise patterns common in AI-generated images.
        """
        if len(residual.shape) == 3:
            residual = np.mean(residual, axis=2)
        
        # Check for grid-like patterns using autocorrelation
        fft = np.fft.fft2(residual)
        power = np.abs(fft)**2
        
        # Look for periodic peaks in power spectrum
        h, w = residual.shape
        center_h, center_w = h // 2, w // 2
        
        # Radial profile
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Check for concentric rings (common in GAN noise)
        radial_mean = []
        for rad in range(1, min(h, w) // 4):
            mask = (r >= rad - 1) & (r < rad)
            if np.any(mask):
                radial_mean.append(np.mean(power[mask]))
        
        if len(radial_mean) < 3:
            return 0.0
        
        # Check for regular peaks in radial profile
        radial_mean = np.array(radial_mean)
        radial_mean = radial_mean / (radial_mean.max() + 1e-10)
        
        # Count peaks
        peaks = 0
        for i in range(1, len(radial_mean) - 1):
            if radial_mean[i] > radial_mean[i-1] and radial_mean[i] > radial_mean[i+1]:
                if radial_mean[i] > 0.1:
                    peaks += 1
        
        # More peaks = more synthetic pattern
        synthetic_score = min(peaks / 10, 1.0)
        
        return float(synthetic_score)
    
    def _analyze_variance_distribution(self, residual: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of noise variance"""
        if len(residual.shape) == 3:
            residual = np.mean(residual, axis=2)
        
        # Compute local variance
        kernel_size = 7
        mean = ndimage.uniform_filter(residual, size=kernel_size)
        mean_sq = ndimage.uniform_filter(residual**2, size=kernel_size)
        local_var = mean_sq - mean**2
        
        # Analyze distribution
        var_flat = local_var.flatten()
        var_flat = var_flat[var_flat > 0]  # Remove zeros
        
        if len(var_flat) == 0:
            return {'variance_anomaly': False}
        
        # Check for bimodal distribution (suspicious)
        hist, bins = np.histogram(var_flat, bins=50)
        hist = hist / hist.sum()
        
        # Count peaks in histogram
        peaks = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                if hist[i] > 0.05:
                    peaks += 1
        
        return {
            'variance_mean': float(np.mean(var_flat)),
            'variance_std': float(np.std(var_flat)),
            'variance_anomaly': peaks > 2  # Bimodal/multimodal = suspicious
        }
    
    def _detect_banding(self, image: np.ndarray) -> float:
        """Detect banding/posterization artifacts"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Normalize
        gray = gray.astype(np.float64)
        if gray.max() > 1:
            gray = gray / 255.0
        
        # Quantize to fewer levels and check for banding
        quantized = np.round(gray * 31) / 31  # 32 levels
        
        # Compute difference
        diff = np.abs(gray - quantized)
        
        # Check for regular patterns in difference
        hist, _ = np.histogram(diff.flatten(), bins=32)
        hist = hist / hist.sum()
        
        # Banding shows as peaks at specific values
        peaks = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                if hist[i] > 0.1:
                    peaks += 1
        
        return float(min(peaks / 5, 1.0))
    
    def _compute_frame_noise_correlation(self, frames: List[np.ndarray]) -> float:
        """Compute noise correlation between consecutive frames"""
        if len(frames) < 2:
            return 0.0
        
        correlations = []
        
        for i in range(len(frames) - 1):
            # Extract noise from both frames
            noise1 = self._extract_noise_residual_2d(
                frames[i].astype(np.float64) / 255.0 if frames[i].max() > 1 else frames[i].astype(np.float64)
            )
            noise2 = self._extract_noise_residual_2d(
                frames[i+1].astype(np.float64) / 255.0 if frames[i+1].max() > 1 else frames[i+1].astype(np.float64)
            )
            
            # Compute correlation
            if noise1.shape != noise2.shape:
                continue
            
            corr = np.corrcoef(noise1.flatten(), noise2.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _extract_audio_noise_floor(self, audio: np.ndarray) -> float:
        """Extract noise floor level in dB"""
        # Use lowest 10% of samples as noise estimate
        sorted_audio = np.sort(np.abs(audio))
        noise_samples = sorted_audio[:len(sorted_audio) // 10]
        
        noise_rms = np.sqrt(np.mean(noise_samples**2))
        
        if noise_rms < 1e-10:
            return -100.0  # Very low noise floor
        
        # Convert to dB
        db = 20 * np.log10(noise_rms + 1e-10)
        
        return float(db)
    
    def _analyze_audio_noise_uniformity(self, audio: np.ndarray) -> float:
        """Analyze uniformity of noise in audio"""
        # Divide into segments
        segment_len = len(audio) // 20
        if segment_len < 100:
            segment_len = 100
        
        variances = []
        for i in range(0, len(audio) - segment_len, segment_len):
            segment = audio[i:i+segment_len]
            variances.append(np.var(segment))
        
        if not variances:
            return 0.0
        
        # Uniformity = low variance of variances
        var_of_vars = np.var(variances)
        mean_var = np.mean(variances)
        
        uniformity = 1.0 / (1.0 + var_of_vars / (mean_var + 1e-10))
        
        return float(uniformity)
    
    def _analyze_audio_noise_spectrum(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze spectral properties of audio noise"""
        try:
            # Compute power spectral density
            freqs, psd = welch(audio, fs=sample_rate, nperseg=min(2048, len(audio)//4))
            
            # Compute spectral flatness
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arithmetic_mean = np.mean(psd)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            # Check for peaks
            psd_norm = psd / (psd.max() + 1e-10)
            peaks = 0
            for i in range(1, len(psd_norm) - 1):
                if psd_norm[i] > psd_norm[i-1] and psd_norm[i] > psd_norm[i+1]:
                    if psd_norm[i] > 0.3:
                        peaks += 1
            
            return {
                'spectral_flatness': float(spectral_flatness),
                'spectral_peaks': int(peaks),
                'spectral_anomaly': spectral_flatness < 0.1 or peaks > 10
            }
        except Exception:
            return {'spectral_anomaly': False}
    
    def _detect_synthetic_audio(self, audio: np.ndarray, sample_rate: int) -> float:
        """Detect synthetic audio characteristics"""
        score = 0.0
        
        # Check for over-regular zero crossings
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        
        if len(zero_crossings) > 1:
            # Compute intervals
            intervals = np.diff(zero_crossings)
            
            # Check regularity
            interval_var = np.var(intervals)
            interval_mean = np.mean(intervals)
            
            regularity = 1.0 / (1.0 + interval_var / (interval_mean + 1e-10))
            
            if regularity > 0.8:
                score += 0.3
        
        # Check for periodicity in amplitude envelope
        envelope = np.abs(audio)
        envelope_smooth = ndimage.uniform_filter(envelope, size=sample_rate // 100)
        
        # FFT of envelope
        envelope_fft = np.abs(np.fft.fft(envelope_smooth))
        envelope_fft = envelope_fft[:len(envelope_fft)//2]
        
        # Check for dominant frequency in envelope
        max_bin = np.argmax(envelope_fft[1:]) + 1  # Skip DC
        max_val = envelope_fft[max_bin]
        
        if max_val > np.mean(envelope_fft) * 3:
            score += 0.2  # Dominant periodicity
        
        return float(min(score, 1.0))
    
    def _analyze_audio_microstructure(self, audio: np.ndarray) -> float:
        """
        Analyze natural microstructure in audio.
        Natural speech has breath, pauses, micro-variations.
        AI speech often lacks these.
        """
        # Compute short-time energy
        window_size = len(audio) // 100
        if window_size < 100:
            window_size = 100
        
        energies = []
        for i in range(0, len(audio) - window_size, window_size):
            segment = audio[i:i+window_size]
            energies.append(np.sqrt(np.mean(segment**2)))
        
        if not energies:
            return 0.0
        
        energies = np.array(energies)
        
        # Natural audio has high dynamic range in energy
        energy_range = np.max(energies) - np.min(energies)
        energy_mean = np.mean(energies)
        
        dynamic_range = energy_range / (energy_mean + 1e-10)
        
        # Also check for silence/pause segments
        silence_threshold = energy_mean * 0.1
        silence_ratio = np.sum(energies < silence_threshold) / len(energies)
        
        # Natural speech typically has 10-40% silence
        natural_silence = 1.0 - abs(silence_ratio - 0.25) * 2
        
        # Combine scores
        microstructure = (min(dynamic_range, 1.0) + max(0, natural_silence)) / 2
        
        return float(microstructure)


def analyze_noise(media: np.ndarray, media_type: str = "auto") -> NoiseResult:
    """Convenience function for noise analysis"""
    analyzer = NoiseAnalyzer()
    return analyzer.analyze(media, media_type)
