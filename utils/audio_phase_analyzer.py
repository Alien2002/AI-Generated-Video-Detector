"""
Audio Phase & Microstructure Analyzer for AI-Generated Content Detection

Analyzes phase coherence and microstructure patterns in audio to detect AI-generated speech.

Key insight: AI-generated audio lacks natural physics-based characteristics:
- Phase coherence: Natural audio has specific phase relationships from vocal tract
- Breath/pause patterns: Natural speech has breathing, pauses, hesitations
- Prosody: Natural speech has rhythm, stress, intonation patterns
- Formant transitions: Smooth transitions between phonemes
- Micro-variations: Subtle timing and pitch variations AI misses

Real audio follows physics of sound production through vocal tract.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging
from scipy.fft import fft

logger = logging.getLogger(__name__)


@dataclass
class AudioPhaseResult:
    """Result of audio phase & microstructure analysis"""
    is_suspicious: bool
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    phase_coherence: float = 0.0
    prosody_score: float = 0.0
    breath_pause_score: float = 0.0
    formant_score: float = 0.0
    microstructure_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class AudioPhaseAnalyzer:
    """
    Analyzes audio phase and microstructure for AI-generated content detection.
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
        # Thresholds
        self.phase_coherence_threshold = 0.85  # High coherence = suspicious
        self.breath_pause_min = 0.05  # Minimum expected breath/pause ratio
        self.prosody_regularity_threshold = 0.7
        
    def analyze_audio(self, audio: np.ndarray, sample_rate: int = None) -> AudioPhaseResult:
        """
        Analyze audio for phase and microstructure characteristics.
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate (optional)
            
        Returns:
            AudioPhaseResult with analysis details
        """
        indicators = []
        confidence = 0.0
        details = {}
        
        if sample_rate:
            self.sample_rate = sample_rate
        
        try:
            # Normalize audio
            audio = audio.flatten().astype(np.float64)
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
            # 1. Phase Coherence Analysis
            phase_result = self._analyze_phase_coherence(audio)
            details['phase_coherence'] = phase_result['coherence']
            details['phase_anomalies'] = phase_result['anomalies']
            
            # 2. Breath & Pause Detection
            breath_result = self._analyze_breath_pauses(audio)
            details['breath_pause_ratio'] = breath_result['ratio']
            details['pause_count'] = breath_result['pause_count']
            details['breath_detected'] = breath_result['breath_detected']
            
            # 3. Prosody Analysis (rhythm, stress, intonation)
            prosody_result = self._analyze_prosody(audio)
            details['prosody_regularity'] = prosody_result['regularity']
            details['pitch_variation'] = prosody_result['pitch_variation']
            details['energy_variation'] = prosody_result['energy_variation']
            
            # 4. Formant Analysis
            formant_result = self._analyze_formants(audio)
            details['formant_transitions'] = formant_result['transitions']
            details['formant_stability'] = formant_result['stability']
            
            # 5. Microstructure Analysis
            micro_result = self._analyze_microstructure(audio)
            details['microstructure_regularity'] = micro_result['regularity']
            details['timing_variation'] = micro_result['timing_variation']
            
            # Generate indicators and compute confidence
            
            # Check 1: Phase Coherence
            if phase_result['coherence'] > self.phase_coherence_threshold:
                indicators.append(f"High phase coherence ({phase_result['coherence']:.3f}) - unnatural for human speech")
                confidence += 0.25
            else:
                indicators.append(f"Phase coherence: {phase_result['coherence']:.3f}")
            
            if phase_result['anomalies'] > 3:
                indicators.append(f"Phase anomalies detected ({phase_result['anomalies']})")
                confidence += 0.1
            
            # Check 2: Breath & Pauses
            if breath_result['ratio'] < self.breath_pause_min:
                indicators.append(f"Missing breath/pause patterns ({breath_result['ratio']:.3f}) - AI speech characteristic")
                confidence += 0.25
            else:
                indicators.append(f"Breath/pause ratio: {breath_result['ratio']:.3f}")
            
            if not breath_result['breath_detected']:
                indicators.append("No breath sounds detected in speech")
                confidence += 0.15
            
            # Check 3: Prosody
            if prosody_result['regularity'] > self.prosody_regularity_threshold:
                indicators.append(f"Over-regular prosody ({prosody_result['regularity']:.3f}) - lacks natural variation")
                confidence += 0.2
            else:
                indicators.append(f"Prosody regularity: {prosody_result['regularity']:.3f}")
            
            if prosody_result['pitch_variation'] < 0.1:
                indicators.append("Low pitch variation - monotone speech pattern")
                confidence += 0.1
            
            # Check 4: Formants
            if formant_result['stability'] > 0.9:
                indicators.append(f"Unusually stable formants ({formant_result['stability']:.3f}) - synthetic speech")
                confidence += 0.15
            
            if formant_result['transitions'] < 0.3:
                indicators.append("Missing smooth formant transitions")
                confidence += 0.1
            
            # Check 5: Microstructure
            if micro_result['regularity'] > 0.8:
                indicators.append(f"Over-regular microstructure ({micro_result['regularity']:.3f}) - lacks natural imperfections")
                confidence += 0.15
            
            if micro_result['timing_variation'] < 0.05:
                indicators.append("Missing natural timing variations")
                confidence += 0.1
            
        except Exception as e:
            logger.error(f"Audio phase analysis error: {e}")
            indicators.append(f"Analysis error: {str(e)}")
        
        confidence = min(confidence, 1.0)
        
        return AudioPhaseResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            phase_coherence=details.get('phase_coherence', 0),
            prosody_score=1.0 - details.get('prosody_regularity', 0),
            breath_pause_score=details.get('breath_pause_ratio', 0),
            formant_score=details.get('formant_transitions', 0),
            microstructure_score=1.0 - details.get('microstructure_regularity', 0),
            details=details
        )
    
    def _analyze_phase_coherence(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze phase coherence in audio signal.
        AI-generated audio often has unnaturally coherent phase.
        """
        # Compute STFT
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Short-time Fourier transform
        stft_matrix = np.zeros((n_fft // 2 + 1, (len(audio) - n_fft) // hop_length + 1), dtype=np.complex128)
        
        for i, start in enumerate(range(0, len(audio) - n_fft, hop_length)):
            segment = audio[start:start + n_fft]
            # Apply Hann window
            windowed = segment * np.hanning(n_fft)
            stft_matrix[:, i] = fft(windowed)[:n_fft // 2 + 1]
        
        # Extract phase
        phases = np.angle(stft_matrix)
        magnitudes = np.abs(stft_matrix)
        
        # Compute phase coherence
        # Natural audio has phase that varies with frequency content
        # AI audio may have more consistent phase patterns
        
        # Phase derivative (phase change over time)
        phase_diff = np.diff(phases, axis=1)
        
        # Wrap phase difference to [-pi, pi]
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        # Compute coherence: low variance in phase derivative = high coherence = suspicious
        # Only consider significant frequency bins
        mag_threshold = np.median(magnitudes)
        significant_mask = magnitudes[:, :-1] > mag_threshold
        
        if np.any(significant_mask):
            phase_var = np.var(phase_diff[significant_mask])
            # Normalize: lower variance = higher coherence
            coherence = 1.0 / (1.0 + phase_var / np.pi)
        else:
            coherence = 0.5
        
        # Count phase anomalies (sudden jumps)
        anomalies = int(np.sum(np.abs(phase_diff) > np.pi * 0.8))
        
        return {
            'coherence': float(coherence),
            'anomalies': int(anomalies)
        }
    
    def _analyze_breath_pauses(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Detect breath and pause patterns in speech.
        Natural speech has breaths and pauses; AI often lacks these.
        """
        # Compute short-time energy
        frame_size = int(self.sample_rate * 0.02)  # 20ms frames
        hop_size = frame_size // 2
        
        energies = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            energies.append(np.sqrt(np.mean(frame**2)))
        
        energies = np.array(energies)
        
        # Normalize energies
        energy_threshold = np.percentile(energies, 20)  # Low energy = silence/pause
        
        # Detect pauses (low energy regions)
        pause_frames = energies < energy_threshold
        
        # Compute pause ratio
        pause_ratio = np.sum(pause_frames) / len(energies)
        
        # Detect breath sounds (brief low-energy with specific spectral content)
        breath_detected = False
        
        # Look for breath patterns: short low-energy bursts
        # Breath sounds typically last 100-300ms
        min_breath_frames = int(0.1 * self.sample_rate / hop_size)
        max_breath_frames = int(0.3 * self.sample_rate / hop_size)
        
        # Find consecutive pause frames
        pause_regions = []
        start_idx = None
        for i, is_pause in enumerate(pause_frames):
            if is_pause and start_idx is None:
                start_idx = i
            elif not is_pause and start_idx is not None:
                pause_regions.append((start_idx, i))
                start_idx = None
        
        if start_idx is not None:
            pause_regions.append((start_idx, len(pause_frames)))
        
        # Count breath-like pauses (appropriate duration)
        breath_count = 0
        for start, end in pause_regions:
            duration_frames = end - start
            if min_breath_frames <= duration_frames <= max_breath_frames:
                breath_count += 1
        
        breath_detected = breath_count > 0
        
        return {
            'ratio': float(pause_ratio),
            'pause_count': len(pause_regions),
            'breath_detected': breath_detected
        }
    
    def _analyze_prosody(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze prosody patterns (rhythm, stress, intonation).
        AI speech often has overly regular prosody.
        """
        # Compute pitch contour using autocorrelation
        frame_size = int(self.sample_rate * 0.03)  # 30ms frames
        hop_size = frame_size // 2
        
        pitches = []
        energies = []
        
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            
            # Compute energy
            energy = np.sqrt(np.mean(frame**2))
            energies.append(energy)
            
            # Compute pitch via autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after lag 0 (fundamental frequency)
            # Voice range: 80-400 Hz
            min_lag = int(self.sample_rate / 400)
            max_lag = int(self.sample_rate / 80)
            
            if max_lag < len(autocorr):
                search_region = autocorr[min_lag:max_lag]
                if len(search_region) > 0 and np.max(search_region) > 0.1 * autocorr[0]:
                    peak_idx = np.argmax(search_region) + min_lag
                    pitch = self.sample_rate / peak_idx
                    pitches.append(pitch)
                else:
                    pitches.append(0)  # Unvoiced
            else:
                pitches.append(0)
        
        pitches = np.array(pitches)
        energies = np.array(energies)
        
        # Compute pitch variation (natural speech has variation)
        voiced_pitches = pitches[pitches > 0]
        if len(voiced_pitches) > 1:
            pitch_variation = np.std(voiced_pitches) / np.mean(voiced_pitches)
        else:
            pitch_variation = 0
        
        # Compute energy variation
        energy_variation = np.std(energies) / (np.mean(energies) + 1e-10)
        
        # Compute regularity: low variation = high regularity = suspicious
        # Natural speech has irregular rhythm
        rhythm_intervals = np.diff(np.where(energies > np.median(energies))[0])
        if len(rhythm_intervals) > 1:
            rhythm_regularity = 1.0 / (1.0 + np.std(rhythm_intervals) / (np.mean(rhythm_intervals) + 1e-10))
        else:
            rhythm_regularity = 0.5
        
        # Combined regularity score
        regularity = (rhythm_regularity + (1.0 - pitch_variation)) / 2
        
        return {
            'regularity': float(regularity),
            'pitch_variation': float(pitch_variation),
            'energy_variation': float(energy_variation)
        }
    
    def _analyze_formants(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze formant patterns.
        Natural speech has smooth formant transitions; AI may have discontinuities.
        """
        # Compute LPC (Linear Predictive Coding) for formant estimation
        frame_size = int(self.sample_rate * 0.03)
        hop_size = frame_size // 2
        lpc_order = 12  # Enough for 4-5 formants
        
        formant_tracks = []
        
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            
            # Apply window
            windowed = frame * np.hamming(len(frame))
            
            try:
                # Compute LPC coefficients
                lpc_coeffs = self._compute_lpc(windowed, lpc_order)
                
                # Find formants from LPC polynomial roots
                roots = np.roots(lpc_coeffs)
                
                # Keep only roots with positive imaginary part (stable)
                roots = roots[np.imag(roots) >= 0]
                
                # Convert to frequencies
                angles = np.angle(roots)
                freqs = angles * self.sample_rate / (2 * np.pi)
                
                # Keep frequencies in voice range (0-5000 Hz)
                freqs = freqs[(freqs > 0) & (freqs < 5000)]
                freqs = np.sort(freqs)[:4]  # First 4 formants
                
                if len(freqs) >= 2:
                    formant_tracks.append(freqs[:2])  # F1, F2
            except Exception:
                pass
        
        if not formant_tracks:
            return {'transitions': 0.5, 'stability': 0.5}
        
        formant_tracks = np.array(formant_tracks)
        
        # Analyze formant transitions
        # Natural speech: smooth transitions
        # AI speech: may have jumps or too stable
        
        # Compute formant velocity (change rate)
        f1_velocity = np.abs(np.diff(formant_tracks[:, 0]))
        f2_velocity = np.abs(np.diff(formant_tracks[:, 1]))
        
        # Transition score: moderate velocity is good
        # Too high = discontinuities, too low = too stable
        avg_velocity = (np.mean(f1_velocity) + np.mean(f2_velocity)) / 2
        
        # Natural speech has average formant velocity around 50-200 Hz per frame
        transitions = 1.0 - abs(avg_velocity - 100) / 200
        transitions = max(0, min(1, transitions))
        
        # Stability: low variance in formant positions = suspicious
        f1_var = np.var(formant_tracks[:, 0])
        f2_var = np.var(formant_tracks[:, 1])
        
        # Normalize by mean
        f1_stability = 1.0 / (1.0 + f1_var / (np.mean(formant_tracks[:, 0])**2 + 1e-10))
        f2_stability = 1.0 / (1.0 + f2_var / (np.mean(formant_tracks[:, 1])**2 + 1e-10))
        
        stability = (f1_stability + f2_stability) / 2
        
        return {
            'transitions': float(transitions),
            'stability': float(stability)
        }
    
    def _compute_lpc(self, frame: np.ndarray, order: int) -> np.ndarray:
        """Compute LPC coefficients using autocorrelation method"""
        # Compute autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Levinson-Durbin recursion
        a = np.zeros(order + 1)
        a[0] = 1.0
        
        error = autocorr[0]
        
        for i in range(1, order + 1):
            # Compute reflection coefficient
            reflection = -np.sum(autocorr[1:i+1] * a[:i][::-1]) / (error + 1e-10)
            
            # Update coefficients
            a[i] = reflection
            for j in range(1, i):
                a[j] = a[j] + reflection * a[i - j]
            
            # Update error
            error = error * (1 - reflection**2)
        
        return a
    
    def _analyze_microstructure(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze microstructure patterns.
        Natural speech has subtle timing and amplitude variations.
        """
        # Analyze fine timing structure
        frame_size = int(self.sample_rate * 0.01)  # 10ms frames
        hop_size = frame_size // 2
        
        # Compute zero-crossing rate (microstructure indicator)
        zcr = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            crossings = np.sum(np.abs(np.diff(np.signbit(frame))))
            zcr.append(crossings / len(frame))
        
        zcr = np.array(zcr)
        
        # Analyze timing regularity
        # Natural speech has jitter (small timing variations)
        zcr_diff = np.diff(zcr)
        timing_variation = np.std(zcr_diff) / (np.mean(zcr) + 1e-10)
        
        # Regularity: low variation = high regularity = suspicious
        regularity = 1.0 / (1.0 + timing_variation * 10)
        
        return {
            'regularity': float(regularity),
            'timing_variation': float(timing_variation)
        }


def analyze_audio_phase(audio: np.ndarray, sample_rate: int = 22050) -> AudioPhaseResult:
    """Convenience function for audio phase analysis"""
    analyzer = AudioPhaseAnalyzer(sample_rate)
    return analyzer.analyze_audio(audio, sample_rate)
