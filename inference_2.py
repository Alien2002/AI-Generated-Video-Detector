import cv2
import clip
import re
import argparse
import numpy as np
import onnx
import torch
from models.TMC import ETMC
from models import image
from PIL import Image
from utils.transformers_visual_detector import TransformersVisualDetector
from utils.category_breakdown import build_image_categories, categories_to_dict
from utils.report_renderer import render_detection_report_html
from utils.calibration import load_calibrator
from utils.ocr_analyzer import analyze_text_ocr
from utils.metadata_analyzer import analyze_metadata, get_metadata_summary
from utils.frequency_analyzer import analyze_frequency
from utils.prnu_analyzer import analyze_prnu
from utils.cfa_analyzer import analyze_cfa
from utils.noise_analyzer import analyze_noise
from utils.temporal_analyzer import analyze_temporal
from utils.audio_phase_analyzer import analyze_audio_phase
from utils.fusion_analyzer import fuse_image_analysis, fuse_video_analysis, fuse_audio_analysis

from onnx2pytorch import ConvertModel
from facenet_pytorch import MTCNN
from models.genconvit_onnx import load_genconvit_onnx


def _load_audio_mono(input_audio_path):
    """Load audio as mono float32 numpy array using pydub (avoids librosa dependency)."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(input_audio_path)
    seg = seg.set_channels(1)
    sr = int(seg.frame_rate)
    samples = np.array(seg.get_array_of_samples())
    # Convert to float32 in [-1, 1]
    max_val = float(1 << (8 * seg.sample_width - 1))
    audio = samples.astype(np.float32) / max_val
    return audio, sr

# BLIP for scene understanding
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print("blip successfully loaded..")
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: transformers not installed. Scene captioning disabled. Install with: pip install transformers")

# Initialize GenConViT ONNX models
genconvit = load_genconvit_onnx('checkpoints/genconvit_weights', use_ed=True, use_vae=True)

# Initialize MTCNN with more sensitive thresholds for better face detection
mtcnn = MTCNN(
    keep_all=True, 
    device='cpu',
    thresholds=[0.5, 0.6, 0.6],  # Lower thresholds to catch more faces (default is [0.6, 0.7, 0.7])
    factor=0.709,  # Scale factor for image pyramid
    min_face_size=20  # Minimum face size in pixels
)

onnx_model = onnx.load('checkpoints/efficientnet.onnx')
pytorch_model = ConvertModel(onnx_model)

#Set random seed for reproducibility.
torch.manual_seed(42)


_TRANSFORMERS_VISUAL_DETECTOR = None
_PLATT_CALIBRATOR = None


def _get_transformers_visual_detector():
    global _TRANSFORMERS_VISUAL_DETECTOR
    if _TRANSFORMERS_VISUAL_DETECTOR is None:
        _TRANSFORMERS_VISUAL_DETECTOR = TransformersVisualDetector(
            model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
            cache_dir="checkpoints/hf_cache",
            device="cpu",
        )
    return _TRANSFORMERS_VISUAL_DETECTOR


def _get_platt_calibrator():
    global _PLATT_CALIBRATOR
    if _PLATT_CALIBRATOR is None:
        _PLATT_CALIBRATOR = load_calibrator("checkpoints/calibration/platt_fusion.json")
    return _PLATT_CALIBRATOR


def _calibrate_probability(p: float) -> float:
    cal = _get_platt_calibrator()
    if cal is None:
        return float(p)
    try:
        return float(cal.predict_proba(float(p)))
    except Exception:
        return float(p)


# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_classes': 2
}


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="datasets/train/fakeavceleb*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=1024)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MMDF")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default = False)
    parser.add_argument("--freeze_image_encoder", type=bool, default = False)
    parser.add_argument("--pretrained_audio_encoder", type = bool, default=False)
    parser.add_argument("--freeze_audio_encoder", type = bool, default = False)
    parser.add_argument("--augment_dataset", type = bool, default = True)

    for key, value in audio_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

def model_summary(args):
    '''Prints the model summary.'''
    model = ETMC(args)

    for name, layer in model.named_modules():
        print(name, layer)

def load_multimodal_model(args):
    '''Load multimodal model'''
    model = ETMC(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt, strict = True)
    model.eval()
    return model

def load_img_modality_model(args):
    '''Loads image modality model.'''
    rgb_encoder = pytorch_model

    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'), weights_only=False)
    rgb_encoder.load_state_dict(ckpt['rgb_encoder'], strict = True)
    rgb_encoder.eval()
    return rgb_encoder

def load_spec_modality_model(args):
    spec_encoder = image.RawNet(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'), weights_only=False)
    spec_encoder.load_state_dict(ckpt['spec_encoder'], strict = True)
    spec_encoder.eval()
    return spec_encoder


#Load models.
parser = argparse.ArgumentParser(description="Inference models")
get_args(parser)
args, remaining_args = parser.parse_known_args()
assert remaining_args == [], remaining_args

spec_model = load_spec_modality_model(args)

img_model = load_img_modality_model(args)
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
clip_text_tokens = clip.tokenize(["a real human face", "an AI generated face"])

# Initialize BLIP-2 for scene understanding
blip_processor = None
blip_model = None
if BLIP_AVAILABLE:
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()
        print("BLIP-2 loaded successfully for scene captioning")
    except Exception as e:
        print(f"Warning: Could not load BLIP-2: {e}")
        BLIP_AVAILABLE = False

# Artifact descriptions for detailed analysis - expanded with multiple descriptions per category
ARTIFACT_DESCRIPTIONS = {
    "skin_texture": [
        "a face with natural skin pores and fine texture details",
        "a face with overly smooth porcelain skin lacking pores",
        "a face with waxy artificial plastic-looking skin",
        "a face with blurred skin details like an oil painting",
    ],
    "eye_reflection": [
        "a face with natural catchlights and reflections in the eyes",
        "a face with inconsistent or missing eye reflections",
        "a face with unnaturally symmetric perfect eye shine",
        "a face with dead flat eyes lacking natural sparkle",
    ],
    "hair_details": [
        "a face with natural individual distinct hair strands",
        "a face with blurry merged hair clumps without definition",
        "a face with hair that looks painted or artificially rendered",
        "a face with hair strands merging into the background",
    ],
    "facial_proportions": [
        "a face with natural proportional balanced features",
        "a face with slightly distorted asymmetric facial proportions",
        "a face with mismatched eye sizes or unnatural spacing",
        "a face with misaligned features or unnatural jawline",
    ],
    "background_text": [
        "a photo with sharp readable text in the background",
        "a photo with garbled blurry nonsense text that makes no sense",
        "a photo with distorted smeared letters in signs or backgrounds",
        "a photo with letters blending together unnaturally",
    ],
    "teeth_tongue": [
        "a face with natural teeth and tongue definition",
        "a face with blurry undefined teeth or extra teeth",
        "a face with teeth that look merged or oddly shaped",
        "a face with tongue lacking texture or definition",
    ],
    "ear_details": [
        "a face with detailed natural ears showing cartilage structure",
        "a face with simplified or distorted ear structure",
        "a face with ears lacking natural detail or too smooth",
        "a face with asymmetric or misshapen ears",
    ],
    "glass_reflection": [
        "eyeglasses with natural consistent realistic reflections",
        "eyeglasses with unnatural inconsistent reflections",
        "eyeglasses with missing reflections or too dark lenses",
        "eyeglasses with reflections that don't match the scene",
    ],
    "finger_hands": [
        "hands with natural fingers showing proper anatomy",
        "hands with distorted fingers extra digits or missing fingers",
        "hands with blurry merged fingers lacking definition",
        "hands with unnatural proportions or wrong number of fingers",
    ],
    "overall_quality": [
        "a photo with natural lighting and minor imperfections",
        "a photo with overly perfect flawless artificial appearance",
        "a photo with inconsistent lighting on different face parts",
        "a photo with unnatural color saturation or HDR effect",
    ],
}


def analyze_artifacts_clip(face_crop, top_n=4):
    """Use CLIP to analyze specific visual artifacts in a face with dynamic ranking."""
    face_pil = Image.fromarray(face_crop.astype(np.uint8))
    face_clip = clip_preprocess(face_pil).unsqueeze(0)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(face_clip)
    
    # Score all descriptions by similarity
    all_scores = []
    
    for category, descriptions in ARTIFACT_DESCRIPTIONS.items():
        for desc in descriptions:
            desc_tokens = clip.tokenize([desc])
            
            with torch.no_grad():
                desc_features = clip_model.encode_text(desc_tokens)
            
            # Calculate cosine similarity
            similarity = (100.0 * image_features @ desc_features.T).softmax(dim=-1)[0, 0].item()
            
            # Determine if this suggests real or fake
            # First description in each category is the "natural/real" one
            is_real_indicator = (descriptions.index(desc) == 0)
            
            all_scores.append({
                'category': category,
                'description': desc,
                'similarity': similarity,
                'is_real_indicator': is_real_indicator,
                'clean_desc': desc.replace("a face with ", "").replace("a photo with ", "").replace("hands with ", "").replace("eyeglasses with ", "")
            })
    
    # Sort by similarity score
    all_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Get top N matches
    top_matches = all_scores[:top_n]
    
    # Separate into real and fake indicators based on what matched
    detected_issues = [(m['category'], m['clean_desc']) for m in top_matches if not m['is_real_indicator']]
    natural_features = [(m['category'], m['clean_desc']) for m in top_matches if m['is_real_indicator']]
    
    return detected_issues, natural_features, top_matches


def generate_explanation(detected_issues, natural_features, top_matches, is_fake, confidence):
    """Generate concise explanation showing verdict, indicators, and brief assessment."""
    # Shortened indicators (max 2 each)
    fake_list = [desc for _, desc in detected_issues[:2]]
    real_list = [desc for _, desc in natural_features[:2]]
    
    fake_text = " | ".join(fake_list) if fake_list else "None major"
    real_text = " | ".join(real_list) if real_list else "None major"
    
    explanation = f"""VERDICT: Likely {"FAKE" if is_fake else "REAL"} ({confidence}% confidence)

⚠️  Fake indicators: {fake_text}
✓  Real indicators: {real_text}

Summary: {"Synthetic artifacts dominate" if is_fake else "Natural features dominate"}. {"Some natural elements present but outweighed by artifacts." if is_fake else "Minor artifacts present but consistent with authentic content."}"""
    
    return explanation


def _generate_image_analysis_summary(final_verdict: str, categories: list) -> str:
    top = sorted(categories, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:3]
    top_bits = []
    for c in top:
        name = c.get("name", "")
        score = float(c.get("score", 0.0))
        top_bits.append(f"{name} ({score*100:.0f}%)")

    if final_verdict == "AI_GENERATED":
        return (
            "The image shows multiple indicators consistent with AI generation. "
            f"Strongest signals: {', '.join(top_bits)}."
        )
    if final_verdict == "REAL":
        return (
            "The image appears largely authentic with limited synthetic artifacts. "
            f"Most notable checks: {', '.join(top_bits)}."
        )
    return (
        "The analysis shows mixed signals. Some detectors indicate AI-like artifacts, while others suggest camera authenticity. "
        f"Top contributing checks: {', '.join(top_bits)}."
    )


def deepfakes_image_predict_report(input_image_path):
    """Return a deepfakedetection.io-style HTML report for images."""
    if not isinstance(input_image_path, str):
        # Expect filepath from Gradio; fallback to existing text pipeline.
        return deepfakes_image_predict_with_metadata(input_image_path)

    img = Image.open(input_image_path)
    img_array = np.array(img)

    # Pretrained transformers visual probability (patch-sampled)
    visual_probability = None
    try:
        detector = _get_transformers_visual_detector()
        visual_probability = detector.predict_deepfake_probability(img_array)
        print(f"[VISUAL-TF] deepfake_probability={visual_probability:.3f}")
    except Exception as e:
        print(f"[VISUAL-TF] unavailable: {e}")

    # Existing analyzers
    metadata_result = None
    try:
        metadata_result = analyze_metadata(input_image_path)
        print(f"[METADATA] is_suspicious={metadata_result.is_suspicious}, confidence={metadata_result.confidence_score:.2f}")
    except Exception as e:
        print(f"[METADATA] unavailable: {e}")

    freq_result = None
    try:
        freq_result = analyze_frequency(img_array)
        print(f"[FREQUENCY] is_suspicious={freq_result.is_suspicious}, confidence={freq_result.confidence_score:.2f}")
    except Exception as e:
        print(f"[FREQUENCY] unavailable: {e}")

    prnu_result = None
    try:
        prnu_result = analyze_prnu(img_array)
        print(f"[PRNU] is_suspicious={prnu_result.is_suspicious}, confidence={prnu_result.confidence_score:.2f}, prnu_present={prnu_result.prnu_present}")
    except Exception as e:
        print(f"[PRNU] unavailable: {e}")

    cfa_result = None
    try:
        cfa_result = analyze_cfa(img_array)
        print(f"[CFA] is_suspicious={cfa_result.is_suspicious}, confidence={cfa_result.confidence_score:.2f}, cfa_detected={cfa_result.cfa_detected}")
    except Exception as e:
        print(f"[CFA] unavailable: {e}")

    noise_result = None
    try:
        noise_result = analyze_noise(img_array, "image")
        print(f"[NOISE] is_suspicious={noise_result.is_suspicious}, confidence={noise_result.confidence_score:.2f}, uniformity={noise_result.noise_uniformity:.3f}")
    except Exception as e:
        print(f"[NOISE] unavailable: {e}")

    ocr_result = None
    try:
        ocr_result = analyze_text_ocr(img_array)
        if ocr_result.metrics:
            print(f"[OCR] text_detected={ocr_result.text_detected}, score={ocr_result.confidence_score:.2f}")
    except Exception as e:
        print(f"[OCR] unavailable: {e}")

    # Wrap visual_probability into Fusion-compatible object
    visual_obj = None
    if visual_probability is not None:
        class _VisualScore:
            def __init__(self, p: float):
                self.confidence_score = float(p)
                self.is_suspicious = bool(p >= 0.5)
                self.indicators = []

        visual_obj = _VisualScore(float(visual_probability))

    fusion_result = None
    try:
        fusion_result = fuse_image_analysis(
            visual_result=visual_obj,
            metadata_result=metadata_result,
            frequency_result=freq_result,
            prnu_result=prnu_result,
            cfa_result=cfa_result,
            noise_result=noise_result,
        )
        print(
            f"[FUSION] verdict={fusion_result.final_verdict}, confidence={fusion_result.confidence:.2f}, probability={fusion_result.calibrated_probability:.2f}"
        )
    except Exception as e:
        print(f"[FUSION] unavailable: {e}")

    categories = build_image_categories(
        image=img_array,
        ocr_result=ocr_result,
        frequency_result=freq_result,
        noise_result=noise_result,
        prnu_result=prnu_result,
        cfa_result=cfa_result,
        metadata_result=metadata_result,
        visual_probability=visual_probability,
    )
    categories_dict = categories_to_dict(categories)

    overall_score = float(fusion_result.calibrated_probability) if fusion_result is not None else float(visual_probability or 0.5)
    if fusion_result is not None:
        overall_verdict = fusion_result.final_verdict
    else:
        overall_verdict = "UNCERTAIN"

    summary = _generate_image_analysis_summary(overall_verdict, categories_dict)

    report = {
        "overall_forgery_score": overall_score,
        "final_verdict": overall_verdict,
        "analysis_summary": summary,
        "categories": categories_dict,
        "signals": {
            "visual_probability": visual_probability,
            "fusion": fusion_result.details if fusion_result is not None else {},
        },
    }

    return render_detection_report_html(report)


def deepfakes_video_predict_report(input_video_path):
    """Return a deepfakedetection.io-style HTML report for videos."""
    if not isinstance(input_video_path, str):
        return deepfakes_video_predict_with_metadata(input_video_path)

    # Sample frames for visual transformer detector and noise analysis
    cap = cv2.VideoCapture(input_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idxs = np.linspace(0, max(0, total - 1), 8).astype(int) if total > 0 else np.array([], dtype=int)
    frames = []
    cur = 0
    idx_set = set(int(i) for i in frame_idxs.tolist())
    while True:
        ret = cap.grab()
        if not ret:
            break
        if cur in idx_set:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur += 1
        if len(frames) >= 8:
            break
    cap.release()

    visual_probability = None
    try:
        detector = _get_transformers_visual_detector()
        if frames:
            probs = [detector.predict_deepfake_probability(f) for f in frames]
            visual_probability = float(0.7 * float(np.mean(probs)) + 0.3 * float(np.max(probs)))
        print(f"[VISUAL-TF-VIDEO] deepfake_probability={visual_probability}")
    except Exception as e:
        print(f"[VISUAL-TF-VIDEO] unavailable: {e}")

    metadata_result = None
    try:
        metadata_result = analyze_metadata(input_video_path)
    except Exception as e:
        print(f"[METADATA] unavailable: {e}")

    noise_result = None
    try:
        if frames:
            noise_result = analyze_noise(np.array(frames), "video")
    except Exception as e:
        print(f"[NOISE-VIDEO] unavailable: {e}")

    temporal_result = None
    try:
        temporal_result = analyze_temporal(input_video_path)
    except Exception as e:
        print(f"[TEMPORAL] unavailable: {e}")

    visual_obj = None
    if visual_probability is not None:
        class _VisualScore:
            def __init__(self, p: float):
                self.confidence_score = float(p)
                self.is_suspicious = bool(p >= 0.5)
                self.indicators = []

        visual_obj = _VisualScore(float(visual_probability))

    fusion_result = None
    try:
        fusion_result = fuse_video_analysis(
            visual_result=visual_obj,
            metadata_result=metadata_result,
            temporal_result=temporal_result,
            noise_result=noise_result,
        )
    except Exception as e:
        print(f"[FUSION-VIDEO] unavailable: {e}")

    overall_score = float(fusion_result.calibrated_probability) if fusion_result is not None else float(visual_probability or 0.5)
    overall_score = _calibrate_probability(overall_score)
    final_verdict = fusion_result.final_verdict if fusion_result is not None else "UNCERTAIN"

    categories = []
    if temporal_result is not None:
        categories.append({
            "name": "Temporal Consistency",
            "score": float(getattr(temporal_result, "confidence_score", 0.0)),
            "description": "Checks motion/identity consistency across frames (optical flow, drift, transitions).",
            "metrics": {
                "flow_consistency": float(getattr(temporal_result, "flow_consistency", 0.0)),
                "identity_stability": float(getattr(temporal_result, "identity_stability", 0.0)),
            },
        })
    if noise_result is not None:
        categories.append({
            "name": "Noise Realism",
            "score": float(getattr(noise_result, "confidence_score", 0.0)),
            "description": "Evaluates whether temporal noise patterns are consistent with real capture.",
            "metrics": {"confidence": float(getattr(noise_result, "confidence_score", 0.0))},
        })
    if metadata_result is not None:
        categories.append({
            "name": "Metadata / Provenance",
            "score": float(getattr(metadata_result, "confidence_score", 0.0)),
            "description": "Reviews container metadata for provenance clues.",
            "metrics": {"indicators": (getattr(metadata_result, "indicators", []) or [])[:3]},
        })
    if visual_probability is not None:
        categories.append({
            "name": "Learned Visual Detector",
            "score": float(visual_probability),
            "description": "Pretrained frame-based detector (patch sampled) aggregated over sampled frames.",
            "metrics": {"visual_probability": float(visual_probability)},
        })

    summary = _generate_image_analysis_summary(final_verdict, categories)
    report = {
        "overall_forgery_score": overall_score,
        "final_verdict": final_verdict,
        "analysis_summary": summary,
        "categories": categories,
        "signals": {"fusion": fusion_result.details if fusion_result is not None else {}},
    }
    return render_detection_report_html(report)


def deepfakes_audio_predict_report(input_audio_path):
    """Return a deepfakedetection.io-style HTML report for audio."""
    audio_result = deepfakes_spec_predict(input_audio_path)

    metadata_result = None
    if isinstance(input_audio_path, str):
        try:
            metadata_result = analyze_metadata(input_audio_path)
        except Exception as e:
            print(f"[METADATA-AUDIO] unavailable: {e}")

    # Noise + phase results (reuse existing analyzers)
    noise_result = None
    audio = None
    sr = 22050
    try:
        if isinstance(input_audio_path, str):
            audio, sr = _load_audio_mono(input_audio_path)
        else:
            audio = input_audio_path[0] if isinstance(input_audio_path, tuple) else input_audio_path
        noise_result = analyze_noise(audio, "audio")
    except Exception as e:
        print(f"[NOISE-AUDIO] unavailable: {e}")

    phase_result = None
    try:
        if audio is not None:
            phase_result = analyze_audio_phase(audio, sr)
    except Exception as e:
        print(f"[AUDIO-PHASE] unavailable: {e}")

    # Convert spectral string verdict into probability-like score
    spectral_obj = extract_visual_confidence(audio_result)

    fusion_result = None
    try:
        fusion_result = fuse_audio_analysis(
            spectral_result=spectral_obj,
            metadata_result=metadata_result,
            noise_result=noise_result,
            phase_result=phase_result,
        )
    except Exception as e:
        print(f"[FUSION-AUDIO] unavailable: {e}")

    overall_score = float(fusion_result.calibrated_probability) if fusion_result is not None else float(spectral_obj.confidence_score)
    overall_score = _calibrate_probability(overall_score)
    final_verdict = fusion_result.final_verdict if fusion_result is not None else "UNCERTAIN"

    categories = []
    categories.append({
        "name": "Spectral Model",
        "score": float(spectral_obj.confidence_score),
        "description": "Neural detector operating on audio spectral patterns.",
        "metrics": {},
    })
    if phase_result is not None:
        categories.append({
            "name": "Phase / Microstructure",
            "score": float(getattr(phase_result, "confidence_score", 0.0)),
            "description": "Checks phase coherence, prosody and micro-variation patterns that can indicate synthesized speech.",
            "metrics": {
                "phase_coherence": float(getattr(phase_result, "phase_coherence", 0.0)),
                "prosody_score": float(getattr(phase_result, "prosody_score", 0.0)),
            },
        })
    if noise_result is not None:
        categories.append({
            "name": "Noise Realism",
            "score": float(getattr(noise_result, "confidence_score", 0.0)),
            "description": "Evaluates whether audio noise/residual structure resembles real recordings.",
            "metrics": {},
        })
    if metadata_result is not None:
        categories.append({
            "name": "Metadata / Provenance",
            "score": float(getattr(metadata_result, "confidence_score", 0.0)),
            "description": "Reviews file/container metadata for provenance clues.",
            "metrics": {"indicators": (getattr(metadata_result, "indicators", []) or [])[:3]},
        })

    summary = _generate_image_analysis_summary(final_verdict, categories)
    report = {
        "overall_forgery_score": overall_score,
        "final_verdict": final_verdict,
        "analysis_summary": summary,
        "categories": categories,
        "signals": {"fusion": fusion_result.details if fusion_result is not None else {}},
    }
    return render_detection_report_html(report)


def preprocess_img(face):
    """Preprocess face with ImageNet normalization for EfficientNet."""
    # Resize to model input size
    face = cv2.resize(face, (224, 224))
    
    # Normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    # ImageNet normalization (mean and std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    face = (face - mean) / std
    
    # Convert to tensor and add batch dimension
    face_pt = torch.unsqueeze(torch.Tensor(face), dim=0)
    return face_pt

def preprocess_for_ensemble(face_crop):
    """Preprocess face for both EfficientNet and CLIP models."""
    # For EfficientNet (your current preprocessing)
    face_efficientnet = preprocess_img(face_crop)
    
    # For CLIP (uses its own preprocessing)
    face_pil = Image.fromarray(face_crop.astype(np.uint8))
    face_clip = clip_preprocess(face_pil).unsqueeze(0)
    
    return face_efficientnet, face_clip

def predict_ensemble_probs(face_crop):
    """Return (real_prob, fake_prob) using EfficientNet+CLIP ensemble."""
    face_eff, face_clip = preprocess_for_ensemble(face_crop)

    with torch.no_grad():
        pred_eff = img_model(face_eff)
        pred_eff = torch.nn.functional.softmax(pred_eff, dim=-1)

    with torch.no_grad():
        image_features = clip_model.encode_image(face_clip)
        text_features = clip_model.encode_text(clip_text_tokens)
        pred_clip = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    weight_eff = 0.6
    weight_clip = 0.4
    ensemble_pred = weight_eff * pred_eff + weight_clip * pred_clip

    real_prob = float(ensemble_pred[0, 0].item())
    fake_prob = float(ensemble_pred[0, 1].item())
    return real_prob, fake_prob

def detect_and_crop_face(input_image):
    """Detect face using MTCNN with fallback strategies."""
    # Convert to numpy if needed
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)
    
    # Ensure RGB format
    if len(input_image.shape) == 2:  # Grayscale
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_image.shape[2] == 3:  # BGR to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    h, w = input_image.shape[:2]
    
    # Try 1: Original image
    boxes, probs = mtcnn.detect(input_image)
    
    # Try 2: If no face, try with slight contrast enhancement
    if boxes is None or len(boxes) == 0:
        lab = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        boxes, probs = mtcnn.detect(enhanced_rgb)
        if boxes is not None and len(boxes) > 0:
            input_image = enhanced_rgb
    
    # Try 3: If still no face, try with resized image (for small faces)
    if boxes is None or len(boxes) == 0:
        scale = 2.0
        large_img = cv2.resize(input_image, (int(w * scale), int(h * scale)))
        boxes, probs = mtcnn.detect(large_img)
        if boxes is not None and len(boxes) > 0:
            # Scale boxes back to original size
            boxes = boxes / scale
            input_image = large_img
    
    # If face found, crop it
    if boxes is not None and len(boxes) > 0:
        best_face_idx = np.argmax(probs)
        box = boxes[best_face_idx].astype(int)
        x1, y1, x2, y2 = box
        
        # Add 30% margin for better context
        margin_x = int((x2 - x1) * 0.3)
        margin_y = int((y2 - y1) * 0.3)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(input_image.shape[1], x2 + margin_x)
        y2 = min(input_image.shape[0], y2 + margin_y)
        
        face_crop = input_image[y1:y2, x1:x2]
        return face_crop, True
    
    # Fallback: Return center crop of the image
    # This handles cases where face detection fails but image might still contain a face
    center_crop_size = min(h, w)
    start_x = (w - center_crop_size) // 2
    start_y = (h - center_crop_size) // 2
    center_crop = input_image[start_y:start_y + center_crop_size, start_x:start_x + center_crop_size]
    return center_crop, False

def preprocess_audio(audio_file):
    audio_pt = torch.unsqueeze(torch.Tensor(audio_file), dim = 0)
    return audio_pt

def deepfakes_spec_predict(input_audio):
    """Audio deepfake detection with detailed spectral analysis explanation."""
    # Handle both file path (from Gradio) and tuple (audio_data, sr)
    if isinstance(input_audio, str):
        # Load audio from file path
        audio_data, _ = _load_audio_mono(input_audio)
        x = audio_data
    elif isinstance(input_audio, tuple):
        x, _ = input_audio
    else:
        x = input_audio
    
    audio = preprocess_audio(x)
    spec_grads = spec_model.forward(audio)
    spec_grads_inv = np.exp(spec_grads.cpu().detach().numpy().squeeze())

    max_value = np.argmax(spec_grads_inv)
    confidence_score = float(spec_grads_inv[max_value])
    
    # Determine if fake or real
    is_fake = max_value <= 0.5
    confidence = round((1 - confidence_score if is_fake else confidence_score) * 100, 2)
    
    # Generate detailed explanation based on spectral characteristics
    explanation = generate_audio_explanation(is_fake, confidence, confidence_score)
    
    return explanation


def generate_audio_explanation(is_fake, confidence, raw_score):
    """Generate concise audio analysis explanation."""
    verdict = "FAKE" if is_fake else "REAL"
    
    # Shortened indicators
    fake_short = [
        "Unnatural spectral patterns",
        "Inconsistent pitch/prosody",
        "Vocoder artifacts (metallic/phase issues)",
        "Missing breath sounds"
    ]
    real_short = [
        "Consistent spectral patterns",
        "Natural pitch variation",
        "Natural breath/pause patterns",
        "Realistic phoneme transitions"
    ]
    
    # Select based on confidence
    if is_fake:
        fake_list = fake_short[:max(2, int(confidence/30))]
        real_list = real_short[:max(0, int((100-confidence)/50))] if confidence < 90 else []
    else:
        real_list = real_short[:max(2, int(confidence/30))]
        fake_list = fake_short[:max(0, int((100-confidence)/50))] if confidence < 90 else []
    
    fake_text = " | ".join(fake_list) if fake_list else "None major"
    real_text = " | ".join(real_list) if real_list else "None major"
    
    return f"""VERDICT: Likely {verdict} audio ({confidence}% confidence)

⚠️  Fake indicators: {fake_text}
✓  Real indicators: {real_text}

Summary: {"Vocoder/AI synthesis detected" if is_fake else "Natural human speech characteristics dominate"}. {"Some natural segments but synthetic artifacts prevalent." if is_fake else "Minor artifacts consistent with compression/recording noise."}"""

def deepfakes_image_predict(input_image):
    """Ensemble prediction using EfficientNet + CLIP with detailed artifact analysis."""
    # Detect and crop face first
    face_crop, face_detected = detect_and_crop_face(input_image)

    real_prob, fake_prob = predict_ensemble_probs(face_crop)
    
    # Analyze artifacts with CLIP
    detected_issues, natural_features, top_matches = analyze_artifacts_clip(face_crop)
    
    is_fake = fake_prob > real_prob
    confidence = round(max(real_prob, fake_prob) * 100, 2)
    
    # Generate detailed explanation
    explanation = generate_explanation(detected_issues, natural_features, top_matches, is_fake, confidence)
    
    if face_detected:
        explanation += "\n\nNote: Face was auto-detected in the image."
    
    return explanation


def preprocess_video(input_video, n_frames = 30):
    v_cap = cv2.VideoCapture(input_video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick 'n_frames' evenly spaced frames to sample
    if n_frames is None:
        sample = np.arange(0, v_len)
    else:
        sample = np.linspace(0, v_len - 1, n_frames).astype(int)

    #Loop through frames.
    face_crops = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop, face_detected = detect_and_crop_face(frame)
            if face_crop is None:
                continue
            face_crops.append((face_crop, face_detected))
    v_cap.release()
    return face_crops


def deepfakes_video_predict(input_video):
    '''Perform inference on a video using detailed artifact analysis with scene understanding.'''
    video_face_crops = preprocess_video(input_video)
    if len(video_face_crops) == 0:
        return "No faces detected in the sampled frames. Try a higher-resolution video or better lighting."
    
    # Capture a sample frame for scene captioning (use the first frame with face)
    sample_frame_for_caption = None
    v_cap = cv2.VideoCapture(input_video)
    ret, frame = v_cap.read()
    if ret:
        sample_frame_for_caption = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    v_cap.release()
    
    # Generate scene caption if BLIP is available
    scene_info = None
    if sample_frame_for_caption is not None and BLIP_AVAILABLE:
        caption = generate_scene_caption(sample_frame_for_caption)
        if caption:
            scene_info = extract_scene_objects(caption)
            print(f"Scene detected: {scene_info['caption']} (category: {scene_info['category']})")

    # Analyze each detected face for artifacts
    all_detected_issues = []
    all_natural_features = []
    all_top_matches = []
    real_probs = []
    fake_probs = []

    for face_crop, _ in video_face_crops:
        # Get prediction probabilities
        real_p, fake_p = predict_ensemble_probs(face_crop)
        real_probs.append(real_p)
        fake_probs.append(fake_p)
        
        # Analyze artifacts with CLIP
        detected_issues, natural_features, top_matches = analyze_artifacts_clip(face_crop)
        all_detected_issues.extend(detected_issues)
        all_natural_features.extend(natural_features)
        all_top_matches.extend(top_matches)
    
    # Calculate overall prediction
    real_mean = np.mean(real_probs)
    fake_mean = np.mean(fake_probs)
    is_fake = fake_mean > real_mean
    confidence = round(max(real_mean, fake_mean) * 100, 2)
    
    # Generate contextual explanation with scene understanding
    if scene_info:
        explanation = generate_contextual_video_explanation(
            scene_info, all_detected_issues, all_natural_features, 
            is_fake, confidence, len(video_face_crops)
        )
    else:
        # Fallback to standard explanation without scene context
        explanation = generate_video_explanation_dynamic(
            all_detected_issues, all_natural_features, 
            is_fake, confidence, len(video_face_crops)
        )
    
    return explanation


def generate_video_explanation_dynamic(all_detected_issues, all_natural_features, is_fake, confidence, frame_count):
    """Generate concise video analysis explanation."""
    verdict = "FAKE" if is_fake else "REAL"
    
    # Get unique items (shortened)
    unique_issues = []
    unique_natural = []
    seen = set()
    
    for cat, desc in all_detected_issues:
        if desc not in seen and len(unique_issues) < 2:
            seen.add(desc)
            unique_issues.append(desc)
    
    seen = set()
    for cat, desc in all_natural_features:
        if desc not in seen and len(unique_natural) < 2:
            seen.add(desc)
            unique_natural.append(desc)
    
    fake_text = " | ".join(unique_issues) if unique_issues else "None major"
    real_text = " | ".join(unique_natural) if unique_natural else "None major"
    
    return f"""VERDICT: Likely {verdict} ({confidence}% confidence, {frame_count} frames analyzed)

⚠️  Fake indicators: {fake_text}
✓  Real indicators: {real_text}

Summary: {"Systematic manipulation detected across frames" if is_fake else "Consistent authentic characteristics across frames"}. {"Mixed signals but artifacts prevail." if is_fake else "Minor artifacts but natural features dominate."}"""


def generate_video_explanation(top_issues, is_fake, confidence, frame_count):
    """Generate concise video explanation (fallback)."""
    issues_list = [desc for desc, _ in top_issues[:3]] if top_issues else []
    issues_text = " | ".join(issues_list) if issues_list else "None detected"
    
    if is_fake:
        return f"""VERDICT: Likely FAKE ({confidence}% confidence, {frame_count} frames)

⚠️  Indicators: {issues_text}

Summary: Synthetic generation artifacts detected across analyzed frames."""
    else:
        return f"""VERDICT: Likely REAL ({confidence}% confidence, {frame_count} frames)

✓  Status: No significant AI artifacts detected

Summary: Consistent natural characteristics across all analyzed frames."""


# ==================== BLIP-2 SCENE UNDERSTANDING FUNCTIONS ====================

def generate_scene_caption(frame):
    """Generate a natural language description of the scene using BLIP-2."""
    if not BLIP_AVAILABLE or blip_processor is None or blip_model is None:
        return None
    
    try:
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame.astype(np.uint8))
        else:
            frame_pil = frame
        
        # Generate caption
        inputs = blip_processor(frame_pil, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=50, num_beams=3)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Warning: Could not generate scene caption: {e}")
        return None


def extract_scene_objects(caption):
    """Extract key objects and activity context from a scene caption."""
    if caption is None:
        return ["person"]
    
    caption_lower = caption.lower()
    
    # Activity/scene categories with keywords
    scene_categories = {
        "sports": ["player", "soccer", "football", "basketball", "tennis", "running", "game", "match", "sport", "ball", "field", "court", "stadium"],
        "interview": ["interview", "talking", "speaking", "conversation", "discussion", "microphone", "news anchor", "presenter"],
        "presentation": ["presentation", "lecture", "speech", "talking", "standing", "audience", "stage", "podium"],
        "performance": ["singing", "dancing", "performance", "concert", "stage", "musician", "band", "singer"],
        "cooking": ["cooking", "kitchen", "chef", "food", "recipe", "preparing", "cutting", "stove"],
        "outdoor": ["outdoor", "street", "park", "beach", "nature", "walking", "outside"],
        "indoor": ["indoor", "room", "office", "house", "living room", "bedroom", "studio"],
        "action": ["action", "fighting", "running", "jumping", "explosion", "chase", "movie scene"]
    }
    
    # Detect category
    detected_category = None
    for category, keywords in scene_categories.items():
        if any(kw in caption_lower for kw in keywords):
            detected_category = category
            break
    
    # Extract key nouns (objects/people) - simple keyword matching
    key_objects = []
    
    object_keywords = {
        "person": ["person", "man", "woman", "people", "player", "child", "girl", "boy", "individual"],
        "face": ["face", "facial"],
        "hands": ["hands", "hand", "fingers", "holding"],
        "background": ["background", "scenery", "environment", "setting"],
        "ball": ["ball", "football", "soccer ball", "basketball"],
        "equipment": ["microphone", "camera", "phone", "laptop", "computer"],
        "clothing": ["clothing", "shirt", "dress", "uniform", "jacket"]
    }
    
    for obj_type, keywords in object_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            key_objects.append(obj_type)
    
    # Ensure we always have at least "person"
    if not key_objects:
        key_objects = ["person"]
    
    return {
        "caption": caption,
        "category": detected_category or "general",
        "objects": key_objects
    }


def contextualize_artifacts_with_scene(artifacts, scene_info, is_fake):
    """Map detected artifacts to specific objects in the scene context."""
    if scene_info is None or isinstance(scene_info, list):
        # Fallback: just list artifacts generically
        return artifacts
    
    caption = scene_info.get("caption", "")
    category = scene_info.get("category", "general")
    objects = scene_info.get("objects", ["person"])
    
    # Use caption for potential future extensions (context-aware artifact detection)
    _ = caption  # Suppress unused variable warning while keeping for reference
    
    # Artifact-to-object mapping
    artifact_object_map = {
        "facial": ["face", "person"],
        "skin": ["face", "person"],
        "eyes": ["face", "person"],
        "hair": ["person"],
        "teeth": ["person", "face"],
        "ears": ["person", "face"],
        "jawline": ["person", "face"],
        "hands": ["hands", "person"],
        "fingers": ["hands", "person"],
        "background": ["background"],
        "text": ["background"],
        "lighting": ["background", "person"],
        "reflections": ["background", "equipment"],
        "glass": ["equipment", "person"]
    }
    
    contextualized = []
    
    for artifact in artifacts:
        artifact_lower = artifact.lower()
        
        # Find which object this artifact relates to
        related_objects = []
        for art_keyword, obj_list in artifact_object_map.items():
            if art_keyword in artifact_lower:
                related_objects = obj_list
                break
        
        # Match with actual detected objects in scene
        matched_object = None
        for obj in objects:
            if obj in related_objects:
                matched_object = obj
                break
        
        # Generate contextual description
        if matched_object == "person" and category == "sports":
            matched_object = "player"
        elif matched_object == "person" and category == "interview":
            matched_object = "interview subject"
        elif matched_object == "person" and category == "presentation":
            matched_object = "speaker"
        elif matched_object == "person" and category == "performance":
            matched_object = "performer"
        
        if matched_object and matched_object != "person":
            if is_fake:
                contextualized.append(f"The {matched_object}'s {artifact}")
            else:
                contextualized.append(f"The {matched_object}'s natural {artifact}")
        else:
            contextualized.append(artifact)
    
    return contextualized


def generate_contextual_video_explanation(scene_info, detected_issues, natural_features, is_fake, confidence, frame_count):
    """Generate concise scene-aware video explanation."""
    verdict = "FAKE" if is_fake else "REAL"
    
    # Contextualize (shortened)
    issues_ctx = contextualize_artifacts_with_scene([desc for _, desc in detected_issues], scene_info, True)[:2]
    natural_ctx = contextualize_artifacts_with_scene([desc for _, desc in natural_features], scene_info, False)[:2]
    
    fake_text = " | ".join(issues_ctx) if issues_ctx else "None major"
    real_text = " | ".join(natural_ctx) if natural_ctx else "None major"
    
    # Scene line (shortened)
    scene = ""
    if scene_info and isinstance(scene_info, dict) and scene_info.get("caption"):
        scene = f"Scene: {scene_info['caption']}\n"
    
    return f"""{scene}VERDICT: Likely {verdict} ({confidence}% confidence, {frame_count} frames)

⚠️  Fake indicators: {fake_text}
✓  Real indicators: {real_text}

Summary: {"Contextual artifacts suggest manipulation" if is_fake else "Scene context indicates authentic content"}. {"Mixed but synthetic patterns dominate." if is_fake else "Natural features prevalent despite minor issues."}"""


# ==================== METADATA-ENHANCED ANALYSIS FUNCTIONS ====================

class VisualResult:
    """Simple container for visual analysis results"""
    def __init__(self, is_fake: bool, confidence: float):
        self.is_suspicious = is_fake
        self.confidence_score = confidence
        self.indicators = []

def extract_visual_confidence(result_string: str) -> VisualResult:
    """Extract confidence and verdict from visual result string"""
    
    # Default values
    is_fake = False
    confidence = 0.5
    
    # Check for verdict
    if "VERDICT: Likely FAKE" in result_string:
        is_fake = True
    elif "VERDICT: Likely REAL" in result_string:
        is_fake = False
    
    # Extract confidence percentage
    match = re.search(r'\((\d+(?:\.\d+)?)%\s*confidence', result_string)
    if match:
        conf_pct = float(match.group(1)) / 100.0
        # If it's REAL, confidence means "confidence it's real"
        # If it's FAKE, confidence means "confidence it's fake"
        # For fusion, we want confidence_score = P(AI-generated)
        if is_fake:
            confidence = conf_pct
        else:
            confidence = 1.0 - conf_pct
    
    return VisualResult(is_fake, confidence)


def deepfakes_video_predict_with_metadata(input_video):
    """
    Perform video analysis with both visual detection and metadata analysis.
    Returns combined verdict with metadata insights.
    """
    # Get visual prediction
    visual_result = deepfakes_video_predict(input_video)
    
    # Run metadata analysis
    metadata_info = ""
    try:
        metadata_result = analyze_metadata(input_video)
        print(f"[METADATA] is_suspicious={metadata_result.is_suspicious}, confidence={metadata_result.confidence_score:.2f}")
        print(f"[METADATA] indicators={metadata_result.indicators}")
        if metadata_result.warnings:
            print(f"[METADATA] warnings={metadata_result.warnings}")
        if metadata_result.is_suspicious:
            metadata_info = f"""

📋 METADATA ANALYSIS:
{get_metadata_summary(metadata_result)}"""
            # Add warning to verdict if metadata is suspicious
            if "REAL" in visual_result and metadata_result.confidence_score > 0.3:
                visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Metadata suggests AI generation)")
    except Exception as e:
        metadata_info = f"\n📋 Metadata analysis unavailable: {str(e)}"
    
    # Run noise analysis on video frames
    noise_info = ""
    try:
        # Extract some frames for noise analysis
        import cv2
        cap = cv2.VideoCapture(input_video)
        frames = []
        for _ in range(min(10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if frames:
            noise_result = analyze_noise(np.array(frames), "video")
            print(f"[NOISE-VIDEO] is_suspicious={noise_result.is_suspicious}, confidence={noise_result.confidence_score:.2f}")
            if noise_result.indicators:
                print(f"[NOISE-VIDEO] indicators={noise_result.indicators}")
            if noise_result.is_suspicious:
                noise_info = f"""

🔊 VIDEO NOISE ANALYSIS:
Confidence: {noise_result.confidence_score*100:.1f}%
Indicators: {', '.join(noise_result.indicators[:3]) if noise_result.indicators else 'None detected'}"""
                if "REAL" in visual_result and noise_result.confidence_score > 0.35:
                    visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Inconsistent video noise detected)")
    except Exception as e:
        noise_info = f"\n🔊 Video noise analysis unavailable: {str(e)}"
    
    # Run temporal consistency analysis
    temporal_info = ""
    try:
        temporal_result = analyze_temporal(input_video)
        print(f"[TEMPORAL] is_suspicious={temporal_result.is_suspicious}, confidence={temporal_result.confidence_score:.2f}, flow={temporal_result.flow_consistency:.3f}, identity={temporal_result.identity_stability:.3f}")
        if temporal_result.indicators:
            print(f"[TEMPORAL] indicators={temporal_result.indicators}")
        if temporal_result.is_suspicious:
            temporal_info = f"""

⏱️ TEMPORAL CONSISTENCY ANALYSIS:
Confidence: {temporal_result.confidence_score*100:.1f}%
Flow Consistency: {temporal_result.flow_consistency:.3f}
Identity Stability: {temporal_result.identity_stability:.3f}
Indicators: {', '.join(temporal_result.indicators[:3]) if temporal_result.indicators else 'None detected'}"""
            if "REAL" in visual_result and temporal_result.confidence_score > 0.35:
                visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Temporal inconsistencies detected)")
    except Exception as e:
        temporal_info = f"\n⏱️ Temporal analysis unavailable: {str(e)}"
    
    # Run calibrated fusion
    fusion_info = ""
    try:
        # Extract visual confidence for fusion
        visual_confidence = extract_visual_confidence(visual_result)
        
        fusion_result = fuse_video_analysis(
            visual_result=visual_confidence,
            metadata_result=metadata_result if 'metadata_result' in dir() else None,
            temporal_result=temporal_result if 'temporal_result' in dir() else None,
            noise_result=noise_result if 'noise_result' in dir() else None
        )
        print(f"[FUSION-VIDEO] verdict={fusion_result.final_verdict}, confidence={fusion_result.confidence:.2f}")
        
        # Update the visual verdict to match fusion result
        if fusion_result.final_verdict == "AI_GENERATED":
            new_verdict = f"VERDICT: Likely FAKE ({fusion_result.confidence*100:.1f}% confidence)"
            visual_result = re.sub(
                r'VERDICT: Likely (REAL|FAKE).*?\n',
                new_verdict + '\n',
                visual_result,
                count=1
            )
        elif fusion_result.final_verdict == "REAL":
            new_verdict = f"VERDICT: Likely REAL ({fusion_result.confidence*100:.1f}% confidence)"
            visual_result = re.sub(
                r'VERDICT: Likely (REAL|FAKE).*?\n',
                new_verdict + '\n',
                visual_result,
                count=1
            )
        
        fusion_info = f"""

🧮 CALIBRATED FUSION:
Verdict: {fusion_result.final_verdict}
Confidence: {fusion_result.confidence*100:.1f}%
Uncertainty: ±{fusion_result.uncertainty*100:.1f}%
AI Probability: {fusion_result.calibrated_probability*100:.1f}%"""
        if 'forensic_consensus' in fusion_result.details:
            fusion_info += f"\n{fusion_result.details['forensic_consensus']}"""
    except Exception as e:
        fusion_info = f"\n🧮 Fusion analysis unavailable: {str(e)}"
    
    return visual_result + metadata_info + noise_info + temporal_info + fusion_info


def deepfakes_image_predict_with_metadata(input_image_path):
    """
    Perform image analysis with both visual detection and metadata analysis.
    Returns combined verdict with metadata insights.
    """
    # Get visual prediction (input_image_path can be a path or array)
    if isinstance(input_image_path, str):
        # It's a file path - load image and analyze metadata
        img = Image.open(input_image_path)
        img_array = np.array(img)
        visual_result = deepfakes_image_predict(img_array)
        
        # Run metadata analysis
        metadata_info = ""
        try:
            metadata_result = analyze_metadata(input_image_path)
            print(f"[METADATA] is_suspicious={metadata_result.is_suspicious}, confidence={metadata_result.confidence_score:.2f}")
            print(f"[METADATA] indicators={metadata_result.indicators}")
            if metadata_result.warnings:
                print(f"[METADATA] warnings={metadata_result.warnings}")
            if metadata_result.is_suspicious:
                metadata_info = f"""

📋 METADATA ANALYSIS:
{get_metadata_summary(metadata_result)}"""
                if "REAL" in visual_result and metadata_result.confidence_score > 0.3:
                    visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Metadata suggests AI generation)")
        except Exception as e:
            metadata_info = f"\n📋 Metadata analysis unavailable: {str(e)}"
        
        # Run frequency analysis
        freq_info = ""
        try:
            freq_result = analyze_frequency(img_array)
            print(f"[FREQUENCY] is_suspicious={freq_result.is_suspicious}, confidence={freq_result.confidence_score:.2f}")
            if freq_result.indicators:
                print(f"[FREQUENCY] indicators={freq_result.indicators}")
            if freq_result.is_suspicious:
                freq_info = f"""

🔬 FREQUENCY ANALYSIS:
Confidence: {freq_result.confidence_score*100:.1f}%
Indicators: {', '.join(freq_result.indicators[:3]) if freq_result.indicators else 'None detected'}"""
                if "REAL" in visual_result and freq_result.confidence_score > 0.4:
                    visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Frequency patterns suggest AI generation)")
        except Exception as e:
            freq_info = f"\n🔬 Frequency analysis unavailable: {str(e)}"
        
        # Run PRNU analysis (sensor fingerprint detection)
        prnu_info = ""
        try:
            prnu_result = analyze_prnu(img_array)
            print(f"[PRNU] is_suspicious={prnu_result.is_suspicious}, confidence={prnu_result.confidence_score:.2f}, prnu_present={prnu_result.prnu_present}")
            if prnu_result.indicators:
                print(f"[PRNU] indicators={prnu_result.indicators}")
            if prnu_result.is_suspicious:
                prnu_info = f"""

📷 SENSOR FINGERPRINT (PRNU):
Confidence: {prnu_result.confidence_score*100:.1f}%
PRNU Present: {'Yes' if prnu_result.prnu_present else 'No (suspicious)'}
Indicators: {', '.join(prnu_result.indicators[:3]) if prnu_result.indicators else 'None detected'}"""
                if "REAL" in visual_result and prnu_result.confidence_score > 0.35:
                    visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (No camera sensor fingerprint detected)")
        except Exception as e:
            prnu_info = f"\n📷 PRNU analysis unavailable: {str(e)}"
        
        # Run CFA/Demosaicing analysis
        cfa_info = ""
        try:
            cfa_result = analyze_cfa(img_array)
            print(f"[CFA] is_suspicious={cfa_result.is_suspicious}, confidence={cfa_result.confidence_score:.2f}, cfa_detected={cfa_result.cfa_detected}")
            if cfa_result.indicators:
                print(f"[CFA] indicators={cfa_result.indicators}")
            if cfa_result.is_suspicious:
                cfa_info = f"""

🎨 CFA/DEMOSAICING ANALYSIS:
Confidence: {cfa_result.confidence_score*100:.1f}%
CFA Detected: {'Yes (' + cfa_result.bayer_pattern + ')' if cfa_result.cfa_detected else 'No (suspicious)'}
Indicators: {', '.join(cfa_result.indicators[:3]) if cfa_result.indicators else 'None detected'}"""
                if "REAL" in visual_result and cfa_result.confidence_score > 0.35:
                    visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (No Bayer filter artifacts detected)")
        except Exception as e:
            cfa_info = f"\n🎨 CFA analysis unavailable: {str(e)}"
        
        # Run noise residual analysis
        noise_info = ""
        try:
            noise_result = analyze_noise(img_array, "image")
            print(f"[NOISE] is_suspicious={noise_result.is_suspicious}, confidence={noise_result.confidence_score:.2f}, uniformity={noise_result.noise_uniformity:.3f}")
            if noise_result.indicators:
                print(f"[NOISE] indicators={noise_result.indicators}")
            if noise_result.is_suspicious:
                noise_info = f"""

🔊 NOISE RESIDUAL ANALYSIS:
Confidence: {noise_result.confidence_score*100:.1f}%
Uniformity: {noise_result.noise_uniformity:.3f} (high = synthetic)
Indicators: {', '.join(noise_result.indicators[:3]) if noise_result.indicators else 'None detected'}"""
                if "REAL" in visual_result and noise_result.confidence_score > 0.35:
                    visual_result = visual_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Synthetic noise patterns detected)")
        except Exception as e:
            noise_info = f"\n🔊 Noise analysis unavailable: {str(e)}"
        
        # Run calibrated fusion
        fusion_info = ""
        try:
            # Extract visual confidence for fusion
            visual_confidence = extract_visual_confidence(visual_result)
            
            fusion_result = fuse_image_analysis(
                visual_result=visual_confidence,
                metadata_result=metadata_result if 'metadata_result' in dir() else None,
                frequency_result=freq_result if 'freq_result' in dir() else None,
                prnu_result=prnu_result if 'prnu_result' in dir() else None,
                cfa_result=cfa_result if 'cfa_result' in dir() else None,
                noise_result=noise_result if 'noise_result' in dir() else None
            )
            print(f"[FUSION] verdict={fusion_result.final_verdict}, confidence={fusion_result.confidence:.2f}, probability={fusion_result.calibrated_probability:.2f}")
            
            # Update the visual verdict to match fusion result
            if fusion_result.final_verdict == "AI_GENERATED":
                new_verdict = f"VERDICT: Likely FAKE ({fusion_result.confidence*100:.1f}% confidence)"
                # Replace the original verdict line
                visual_result = re.sub(
                    r'VERDICT: Likely (REAL|FAKE).*?\n',
                    new_verdict + '\n',
                    visual_result,
                    count=1
                )
            elif fusion_result.final_verdict == "REAL":
                new_verdict = f"VERDICT: Likely REAL ({fusion_result.confidence*100:.1f}% confidence)"
                visual_result = re.sub(
                    r'VERDICT: Likely (REAL|FAKE).*?\n',
                    new_verdict + '\n',
                    visual_result,
                    count=1
                )
            
            fusion_info = f"""

🧮 CALIBRATED FUSION:
Verdict: {fusion_result.final_verdict}
Confidence: {fusion_result.confidence*100:.1f}%
Uncertainty: ±{fusion_result.uncertainty*100:.1f}%
AI Probability: {fusion_result.calibrated_probability*100:.1f}%"""
            if 'forensic_consensus' in fusion_result.details:
                fusion_info += f"\n{fusion_result.details['forensic_consensus']}"
        except Exception as e:
            fusion_info = f"\n🧮 Fusion analysis unavailable: {str(e)}"
        
        return visual_result + metadata_info + freq_info + prnu_info + cfa_info + noise_info + fusion_info
    else:
        # It's already an array - just run visual prediction with frequency analysis
        visual_result = deepfakes_image_predict(input_image_path)
        
        # Still run frequency analysis on array
        freq_info = ""
        try:
            freq_result = analyze_frequency(input_image_path)
            if freq_result.is_suspicious:
                freq_info = f"""

🔬 FREQUENCY ANALYSIS:
Confidence: {freq_result.confidence_score*100:.1f}%
Indicators: {', '.join(freq_result.indicators[:3]) if freq_result.indicators else 'None detected'}"""
        except Exception:
            pass
        
        return visual_result + freq_info


def deepfakes_audio_predict_with_metadata(input_audio_path):
    """
    Perform audio analysis with both audio detection and metadata analysis.
    Returns combined verdict with metadata insights.
    """
    # Get audio prediction
    audio_result = deepfakes_spec_predict(input_audio_path)
    
    # Run metadata analysis if path is provided
    metadata_info = ""
    if isinstance(input_audio_path, str):
        try:
            metadata_result = analyze_metadata(input_audio_path)
            print(f"[METADATA] is_suspicious={metadata_result.is_suspicious}, confidence={metadata_result.confidence_score:.2f}")
            print(f"[METADATA] indicators={metadata_result.indicators}")
            if metadata_result.warnings:
                print(f"[METADATA] warnings={metadata_result.warnings}")
            if metadata_result.is_suspicious:
                metadata_info = f"""

📋 METADATA ANALYSIS:
{get_metadata_summary(metadata_result)}"""
                if "REAL" in audio_result and metadata_result.confidence_score > 0.3:
                    audio_result = audio_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Metadata suggests AI generation)")
        except Exception as e:
            metadata_info = f"\n📋 Metadata analysis unavailable: {str(e)}"
    
    # Run noise analysis on audio
    noise_info = ""
    try:
        # Load audio for noise analysis
        if isinstance(input_audio_path, str):
            audio, sr = _load_audio_mono(input_audio_path)
        else:
            # It's already audio data
            audio = input_audio_path[0] if isinstance(input_audio_path, tuple) else input_audio_path
            sr = 22050
        
        noise_result = analyze_noise(audio, "audio")
        print(f"[NOISE-AUDIO] is_suspicious={noise_result.is_suspicious}, confidence={noise_result.confidence_score:.2f}")
        if noise_result.indicators:
            print(f"[NOISE-AUDIO] indicators={noise_result.indicators}")
        if noise_result.is_suspicious:
            noise_info = f"""

🔊 AUDIO NOISE ANALYSIS:
Confidence: {noise_result.confidence_score*100:.1f}%
Uniformity: {noise_result.noise_uniformity:.3f} (high = synthetic)
Indicators: {', '.join(noise_result.indicators[:3]) if noise_result.indicators else 'None detected'}"""
            if "REAL" in audio_result and noise_result.confidence_score > 0.35:
                audio_result = audio_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Synthetic audio noise detected)")
    except Exception as e:
        noise_info = f"\n🔊 Audio noise analysis unavailable: {str(e)}"
    
    # Run audio phase & microstructure analysis
    phase_info = ""
    try:
        # Use already loaded audio
        if 'audio' not in dir() or audio is None:
            if isinstance(input_audio_path, str):
                audio, _sr = _load_audio_mono(input_audio_path)
                if 'sr' not in dir():
                    sr = _sr
            else:
                audio = input_audio_path[0] if isinstance(input_audio_path, tuple) else input_audio_path
        
        phase_result = analyze_audio_phase(audio, sr if 'sr' in dir() else 22050)
        print(f"[AUDIO-PHASE] is_suspicious={phase_result.is_suspicious}, confidence={phase_result.confidence_score:.2f}, phase_coherence={phase_result.phase_coherence:.3f}")
        if phase_result.indicators:
            print(f"[AUDIO-PHASE] indicators={phase_result.indicators}")
        if phase_result.is_suspicious:
            phase_info = f"""

🎙️ AUDIO PHASE & MICROSTRUCTURE:
Confidence: {phase_result.confidence_score*100:.1f}%
Phase Coherence: {phase_result.phase_coherence:.3f} (high = synthetic)
Prosody Score: {phase_result.prosody_score:.3f}
Breath/Pause: {phase_result.breath_pause_score:.3f}
Indicators: {', '.join(phase_result.indicators[:3]) if phase_result.indicators else 'None detected'}"""
            if "REAL" in audio_result and phase_result.confidence_score > 0.35:
                audio_result = audio_result.replace("VERDICT: Likely REAL", "VERDICT: Likely REAL ⚠️ (Unnatural audio phase patterns detected)")
    except Exception as e:
        phase_info = f"\n🎙️ Audio phase analysis unavailable: {str(e)}"
    
    # Run calibrated fusion
    fusion_info = ""
    try:
        # Extract audio confidence for fusion
        audio_confidence = extract_visual_confidence(audio_result)
        
        fusion_result = fuse_audio_analysis(
            spectral_result=audio_confidence,
            metadata_result=metadata_result if 'metadata_result' in dir() else None,
            noise_result=noise_result if 'noise_result' in dir() else None,
            phase_result=phase_result if 'phase_result' in dir() else None
        )
        print(f"[FUSION-AUDIO] verdict={fusion_result.final_verdict}, confidence={fusion_result.confidence:.2f}")
        
        # Update the audio verdict to match fusion result
        if fusion_result.final_verdict == "AI_GENERATED":
            new_verdict = f"VERDICT: Likely FAKE audio ({fusion_result.confidence*100:.1f}% confidence)"
            audio_result = re.sub(
                r'VERDICT: Likely (REAL|FAKE).*?\n',
                new_verdict + '\n',
                audio_result,
                count=1
            )
        elif fusion_result.final_verdict == "REAL":
            new_verdict = f"VERDICT: Likely REAL audio ({fusion_result.confidence*100:.1f}% confidence)"
            audio_result = re.sub(
                r'VERDICT: Likely (REAL|FAKE).*?\n',
                new_verdict + '\n',
                audio_result,
                count=1
            )
        
        fusion_info = f"""

🧮 CALIBRATED FUSION:
Verdict: {fusion_result.final_verdict}
Confidence: {fusion_result.confidence*100:.1f}%
Uncertainty: ±{fusion_result.uncertainty*100:.1f}%
AI Probability: {fusion_result.calibrated_probability*100:.1f}%"""
        if 'forensic_consensus' in fusion_result.details:
            fusion_info += f"\n{fusion_result.details['forensic_consensus']}"""
    except Exception as e:
        fusion_info = f"\n🧮 Fusion analysis unavailable: {str(e)}"
    
    return audio_result + metadata_info + noise_info + phase_info + fusion_info

