"""
Calibrated Fusion Analyzer for AI-Generated Content Detection

Combines multiple forensic analysis signals into a unified, calibrated verdict.

Key features:
- Bayesian fusion: Probabilistic combination of evidence
- Logistic regression calibration: Maps raw scores to probabilities
- Uncertainty quantification: Confidence intervals on predictions
- Weighted combination: Different weights for different analyzers

This provides a principled way to combine heterogeneous signals.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of calibrated fusion analysis"""
    final_verdict: str  # 'AI_GENERATED', 'REAL', 'UNCERTAIN'
    confidence: float
    uncertainty: float
    calibrated_probability: float
    individual_scores: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class CalibratedFusion:
    """
    Combines multiple forensic analysis results into calibrated predictions.
    
    Uses Bayesian updating and logistic calibration for robust fusion.
    """
    
    def __init__(self):
        # Pre-defined weights for different analyzers (based on reliability)
        # These can be learned from data, but we use heuristic values
        self.analyzer_weights = {
            # Image analyzers
            'prnu': 0.25,        # Strong physics-based signal - increased
            'cfa': 0.20,         # Strong physics-based signal - increased
            'frequency': 0.18,   # Good for GAN/diffusion detection - increased
            'noise_image': 0.15, # Moderate reliability - increased
            'metadata': 0.07,    # Can be spoofed, lower weight
            'visual': 0.15,      # Deep learning model - reduced, forensics should dominate
            
            # Video analyzers
            'temporal': 0.30,    # Strong for video deepfakes
            'noise_video': 0.20,
            
            # Audio analyzers
            'audio_phase': 0.25, # Strong for AI speech
            'noise_audio': 0.20,
            'spectral': 0.15,
        }
        
        # Calibration parameters (sigmoid parameters)
        # Important: keep sigmoid centered so score=0.5 maps to probability≈0.5.
        # For sigmoid(slope*x + intercept), set intercept≈-slope*0.5.
        self.calibration_params = {
            'prnu': {'slope': 6.0, 'intercept': -3.0},
            'cfa': {'slope': 5.5, 'intercept': -2.75},
            'frequency': {'slope': 5.0, 'intercept': -2.5},
            'noise': {'slope': 5.0, 'intercept': -2.5},
            'temporal': {'slope': 6.0, 'intercept': -3.0},
            'audio_phase': {'slope': 5.5, 'intercept': -2.75},
            'visual': {'slope': 4.0, 'intercept': -2.0},
            'metadata': {'slope': 3.0, 'intercept': -1.5},
        }
        
        # Prior probability of AI-generated content
        self.prior_ai = 0.3  # 30% prior (can be adjusted based on use case)
        
    def fuse_results(self, results: Dict[str, Any], media_type: str = "image") -> FusionResult:
        """
        Fuse multiple analysis results into a calibrated verdict.
        
        Args:
            results: Dictionary of analyzer results
            media_type: 'image', 'video', or 'audio'
            
        Returns:
            FusionResult with combined verdict
        """
        individual_scores = {}
        calibrated_scores = {}
        indicators = []
        details = {}
        
        # Extract and calibrate individual scores
        for analyzer_name, result in results.items():
            if result is None:
                continue
            
            # Get confidence score
            if hasattr(result, 'confidence_score'):
                score = result.confidence_score
            elif hasattr(result, 'is_suspicious'):
                score = 0.7 if result.is_suspicious else 0.2
            else:
                continue
            
            # Store raw score
            individual_scores[analyzer_name] = score
            
            # Calibrate using sigmoid
            cal_params = self._get_calibration_params(analyzer_name)
            calibrated = self._sigmoid(score, cal_params['slope'], cal_params['intercept'])
            calibrated_scores[analyzer_name] = calibrated
            
            # Collect indicators
            if hasattr(result, 'indicators') and result.indicators:
                for ind in result.indicators[:2]:  # Top 2 per analyzer
                    indicators.append(f"[{analyzer_name}] {ind}")
        
        details['raw_scores'] = individual_scores
        details['calibrated_scores'] = calibrated_scores
        
        # Get weights for this media type
        weights = self._get_weights(media_type, list(calibrated_scores.keys()))
        details['weights'] = weights
        
        # Check for forensic consensus (multiple analyzers agreeing)
        forensic_analyzers = ['prnu', 'cfa', 'frequency', 'noise_image', 'noise_video', 'noise_audio', 'temporal', 'audio_phase']
        forensic_scores = [calibrated_scores.get(a, 0.5) for a in forensic_analyzers if a in calibrated_scores]
        
        # Count how many forensic analyzers say AI (score > 0.5)
        forensic_ai_count = sum(1 for s in forensic_scores if s > 0.5)
        forensic_total = len(forensic_scores)
        
        # Consensus bonus: if majority of forensics say AI, slightly boost probability
        consensus_bonus = 0.0
        if forensic_total >= 2:  # Need at least 2 forensic signals
            if forensic_ai_count / forensic_total >= 0.6:  # 60%+ agree on AI
                consensus_bonus = 0.06 * (forensic_ai_count / forensic_total)
                details['consensus_bonus'] = consensus_bonus
                details['forensic_consensus'] = f"{forensic_ai_count}/{forensic_total} forensic signals indicate AI"
        
        # Bayesian fusion
        fused_probability, uncertainty = self._bayesian_fusion(calibrated_scores, weights)
        details['fused_probability'] = fused_probability
        
        # Logistic regression fusion (alternative)
        logistic_prob = self._logistic_fusion(calibrated_scores, weights)
        details['logistic_probability'] = logistic_prob
        
        # Combine both methods (ensemble) + consensus bonus
        final_probability = (fused_probability + logistic_prob) / 2 + consensus_bonus
        final_probability = min(final_probability, 1.0)  # Cap at 1.0
        details['final_probability'] = final_probability
        
        # Determine verdict - stricter thresholds to reduce false positives
        if final_probability > 0.70:
            verdict = "AI_GENERATED"
            confidence = final_probability
        elif final_probability < 0.30:
            verdict = "REAL"
            confidence = 1.0 - final_probability
        else:
            verdict = "UNCERTAIN"
            confidence = 1.0 - abs(final_probability - 0.5) * 2
        
        # Add top indicators
        top_indicators = indicators[:5] if indicators else ["No significant anomalies detected"]
        
        return FusionResult(
            final_verdict=verdict,
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            calibrated_probability=float(final_probability),
            individual_scores=individual_scores,
            weights_used=weights,
            indicators=top_indicators,
            details=details
        )
    
    def fuse_image_results(
        self,
        visual_result: Optional[Any] = None,
        metadata_result: Optional[Any] = None,
        frequency_result: Optional[Any] = None,
        prnu_result: Optional[Any] = None,
        cfa_result: Optional[Any] = None,
        noise_result: Optional[Any] = None
    ) -> FusionResult:
        """Convenience method for fusing image analysis results"""
        results = {}
        
        if visual_result is not None:
            results['visual'] = visual_result
        if metadata_result is not None:
            results['metadata'] = metadata_result
        if frequency_result is not None:
            results['frequency'] = frequency_result
        if prnu_result is not None:
            results['prnu'] = prnu_result
        if cfa_result is not None:
            results['cfa'] = cfa_result
        if noise_result is not None:
            results['noise_image'] = noise_result
        
        return self.fuse_results(results, "image")
    
    def fuse_video_results(
        self,
        visual_result: Optional[Any] = None,
        metadata_result: Optional[Any] = None,
        temporal_result: Optional[Any] = None,
        noise_result: Optional[Any] = None
    ) -> FusionResult:
        """Convenience method for fusing video analysis results"""
        results = {}
        
        if visual_result is not None:
            results['visual'] = visual_result
        if metadata_result is not None:
            results['metadata'] = metadata_result
        if temporal_result is not None:
            results['temporal'] = temporal_result
        if noise_result is not None:
            results['noise_video'] = noise_result
        
        return self.fuse_results(results, "video")
    
    def fuse_audio_results(
        self,
        spectral_result: Optional[Any] = None,
        metadata_result: Optional[Any] = None,
        noise_result: Optional[Any] = None,
        phase_result: Optional[Any] = None
    ) -> FusionResult:
        """Convenience method for fusing audio analysis results"""
        results = {}
        
        if spectral_result is not None:
            results['spectral'] = spectral_result
        if metadata_result is not None:
            results['metadata'] = metadata_result
        if noise_result is not None:
            results['noise_audio'] = noise_result
        if phase_result is not None:
            results['audio_phase'] = phase_result
        
        return self.fuse_results(results, "audio")
    
    def _get_calibration_params(self, analyzer_name: str) -> Dict[str, float]:
        """Get calibration parameters for an analyzer"""
        # Map analyzer names to calibration parameter keys
        name_map = {
            'prnu': 'prnu',
            'cfa': 'cfa',
            'frequency': 'frequency',
            'noise_image': 'noise',
            'noise_video': 'noise',
            'noise_audio': 'noise',
            'temporal': 'temporal',
            'audio_phase': 'audio_phase',
            'visual': 'visual',
            'metadata': 'metadata',
            'spectral': 'audio_phase',  # Similar to audio_phase
        }
        
        key = name_map.get(analyzer_name, 'noise')
        return self.calibration_params.get(key, {'slope': 3.5, 'intercept': -0.8})
    
    def _get_weights(self, media_type: str, available_analyzers: List[str]) -> Dict[str, float]:
        """Get weights for available analyzers, normalized"""
        weights = {}
        
        for analyzer in available_analyzers:
            if analyzer in self.analyzer_weights:
                weights[analyzer] = self.analyzer_weights[analyzer]
            else:
                weights[analyzer] = 0.1  # Default weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _sigmoid(self, x: float, slope: float = 5.0, intercept: float = -1.5) -> float:
        """Sigmoid calibration function"""
        z = slope * x + intercept
        # Clip to avoid overflow
        z = np.clip(z, -20, 20)
        return float(1.0 / (1.0 + np.exp(-z)))
    
    def _bayesian_fusion(
        self, 
        scores: Dict[str, float], 
        weights: Dict[str, float]
    ) -> tuple:
        """
        Bayesian fusion of multiple evidence sources.
        
        Uses log-odds formulation for numerical stability.
        """
        # Prior log-odds
        prior_odds = np.log(self.prior_ai / (1 - self.prior_ai))
        
        # Likelihood ratios for each evidence
        log_likelihoods = []
        
        for analyzer, score in scores.items():
            weight = weights.get(analyzer, 0.1)
            
            # Compute likelihood ratio from score
            # score = P(AI|evidence) / P(evidence)
            # LR = P(evidence|AI) / P(evidence|real)
            
            # Approximate LR from calibrated score
            # If score > 0.5, evidence supports AI
            # LR = score / (1 - score)
            
            eps = 1e-10
            score_clipped = np.clip(score, eps, 1 - eps)
            lr = score_clipped / (1 - score_clipped)
            
            # Apply weight (reduces extreme LR values)
            weighted_lr = np.power(lr, weight)
            
            log_lr = np.log(weighted_lr + eps)
            log_likelihoods.append(log_lr)
        
        # Posterior log-odds
        posterior_log_odds = prior_odds + sum(log_likelihoods)
        
        # Convert to probability
        posterior_prob = 1.0 / (1.0 + np.exp(-posterior_log_odds))
        
        # Compute uncertainty (variance of log-odds)
        # Higher variance = more uncertain
        if len(log_likelihoods) > 1:
            variance = np.var(log_likelihoods)
            uncertainty = 1.0 - np.exp(-variance)  # Map to [0, 1]
        else:
            uncertainty = 0.5
        
        return float(posterior_prob), float(uncertainty)
    
    def _logistic_fusion(
        self, 
        scores: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """
        Logistic regression fusion.
        
        Combines scores as weighted sum through logistic function.
        """
        # Weighted sum of scores
        weighted_sum = 0.0
        total_weight = 0.0
        
        for analyzer, score in scores.items():
            weight = weights.get(analyzer, 0.1)
            weighted_sum += weight * score
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum /= total_weight
        
        # Apply logistic function with calibration
        # Higher scores = more likely AI
        z = 6.0 * (weighted_sum - 0.5)  # Centered at 0.5
        z = np.clip(z, -20, 20)
        probability = 1.0 / (1.0 + np.exp(-z))
        
        return float(probability)
    
    def get_fusion_summary(self, result: FusionResult) -> str:
        """Generate human-readable summary of fusion result"""
        lines = []
        
        # Verdict
        verdict_emoji = {
            'AI_GENERATED': '🤖',
            'REAL': '✅',
            'UNCERTAIN': '❓'
        }
        emoji = verdict_emoji.get(result.final_verdict, '❓')
        
        lines.append(f"\n{emoji} FUSION VERDICT: {result.final_verdict}")
        lines.append(f"   Confidence: {result.confidence*100:.1f}%")
        lines.append(f"   Uncertainty: ±{result.uncertainty*100:.1f}%")
        lines.append(f"   Calibrated Probability: {result.calibrated_probability*100:.1f}%")
        
        # Individual scores
        lines.append("\n📊 Individual Analyzer Scores:")
        for analyzer, score in sorted(result.individual_scores.items(), 
                                     key=lambda x: x[1], reverse=True):
            weight = result.weights_used.get(analyzer, 0)
            lines.append(f"   {analyzer}: {score:.3f} (weight: {weight:.2f})")
        
        # Top indicators
        if result.indicators:
            lines.append("\n🔍 Key Indicators:")
            for ind in result.indicators[:5]:
                lines.append(f"   • {ind}")
        
        return "\n".join(lines)


def fuse_image_analysis(
    visual_result=None,
    metadata_result=None,
    frequency_result=None,
    prnu_result=None,
    cfa_result=None,
    noise_result=None
) -> FusionResult:
    """Convenience function for image fusion"""
    fusion = CalibratedFusion()
    return fusion.fuse_image_results(
        visual_result, metadata_result, frequency_result,
        prnu_result, cfa_result, noise_result
    )


def fuse_video_analysis(
    visual_result=None,
    metadata_result=None,
    temporal_result=None,
    noise_result=None
) -> FusionResult:
    """Convenience function for video fusion"""
    fusion = CalibratedFusion()
    return fusion.fuse_video_results(
        visual_result, metadata_result, temporal_result, noise_result
    )


def fuse_audio_analysis(
    spectral_result=None,
    metadata_result=None,
    noise_result=None,
    phase_result=None
) -> FusionResult:
    """Convenience function for audio fusion"""
    fusion = CalibratedFusion()
    return fusion.fuse_audio_results(
        spectral_result, metadata_result, noise_result, phase_result
    )
