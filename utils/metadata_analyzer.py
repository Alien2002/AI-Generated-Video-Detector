
'''
Metadata Analyzer for AI-Generated Content Detection

Analyzes file metadata to detect potential AI-generated or manipulated content.
Checks for:
- EXIF data inconsistencies
- AI software signatures
- Camera/capture device metadata
- Encoding parameters
- Creation timestamps
'''

import os
import re
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetadataResult:
    """Result of metadata analysis"""
    is_suspicious: bool
    confidence_score: float  # 0.0 to 1.0 (higher = more likely AI-generated)
    indicators: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_details: Dict[str, Any] = field(default_factory=dict)


# Known AI generation tools and their signatures
AI_SOFTWARE_SIGNATURES = {
    # Image generation
    'stable diffusion': ['stable-diffusion', 'stability.ai', 'sdxl', 'sd 1.', 'sd 2.'],
    'midjourney': ['midjourney', 'mj_', 'midjourney_'],
    'dall-e': ['dall-e', 'dalle', 'openai', 'dall·e'],
    'adobe firefly': ['firefly', 'adobe firefly'],
    'leonardo.ai': ['leonardo', 'leonardo.ai'],
    'ideogram': ['ideogram'],
    
    # Video generation
    'runway': ['runway', 'runwayml', 'gen-1', 'gen-2', 'gen-3'],
    'sora': ['sora', 'openai sora'],
    'pika': ['pika', 'pika labs'],
    'synthesia': ['synthesia'],
    'heygen': ['heygen', 'hey-gen'],
    'd-id': ['d-id', 'did.com'],
    
    # Audio generation
    'elevenlabs': ['elevenlabs', 'eleven labs'],
    'murf.ai': ['murf'],
    'descript': ['descript', 'lyrebird'],
    
    # General editing/manipulation tools
    'photoshop': ['adobe photoshop', 'photoshop'],
    'gimp': ['gimp'],
    'after effects': ['after effects', 'aftereffects'],
}

# Suspicious metadata patterns
SUSPICIOUS_PATTERNS = [
    r'generated\s*by',
    r'ai\s*generated',
    r'synthetic',
    r'deepfake',
    r'fake',
    r'artificial',
    r'neural\s*network',
    r'diffusion\s*model',
    r'gan\s*generated',
]

# Legitimate camera manufacturers for validation
LEGITIMATE_CAMERAS = {
    'canon', 'nikon', 'sony', 'fuji', 'fuji film', 'fujifilm',
    'panasonic', 'olympus', 'leica', 'pentax', 'samsung',
    'apple', 'huawei', 'xiaomi', 'google', 'oneplus', 'lg',
    'hasselblad', 'phase one', 'red', 'blackmagic', 'dji',
    'go pro', 'gopro', 'vivo', 'oppo', 'motorola', 'htc',
}


class MetadataAnalyzer:
    """Analyzes file metadata for AI-generation indicators"""
    
    def __init__(self):
        self.ai_signatures = AI_SOFTWARE_SIGNATURES
        self.suspicious_patterns = SUSPICIOUS_PATTERNS
        self.legitimate_cameras = LEGITIMATE_CAMERAS
    
    def analyze_file(self, file_path: str) -> MetadataResult:
        """
        Analyze a file's metadata for AI-generation indicators.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            MetadataResult with analysis details
        """
        if not os.path.exists(file_path):
            return MetadataResult(
                is_suspicious=False,
                confidence_score=0.0,
                warnings=[f"File not found: {file_path}"]
            )
        
        # Determine file type
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']:
            return self._analyze_image(file_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
            return self._analyze_video(file_path)
        elif ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a']:
            return self._analyze_audio(file_path)
        else:
            return MetadataResult(
                is_suspicious=False,
                confidence_score=0.0,
                warnings=[f"Unsupported file type: {ext}"]
            )
    
    def _analyze_image(self, file_path: str) -> MetadataResult:
        """Analyze image metadata"""
        indicators = []
        warnings = []
        raw_metadata = {}
        analysis_details = {}
        confidence = 0.0
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            
            with Image.open(file_path) as img:
                # Get basic image info
                raw_metadata['format'] = img.format
                raw_metadata['size'] = img.size
                raw_metadata['mode'] = img.mode
                
                # Extract EXIF data
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    raw_exif = img._getexif()
                    for tag_id, value in raw_exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'GPSInfo':
                            gps_data = {}
                            for gps_id in value:
                                gps_tag = GPSTAGS.get(gps_id, gps_id)
                                gps_data[gps_tag] = value[gps_id]
                            exif_data['GPSInfo'] = gps_data
                        else:
                            exif_data[tag] = str(value) if value else None
                
                raw_metadata['exif'] = exif_data
                
                # Check for camera info
                camera_info = self._check_camera_info(exif_data)
                analysis_details['camera_check'] = camera_info
                
                if camera_info['missing']:
                    indicators.append("No camera information found in EXIF data")
                    confidence += 0.15
                
                if camera_info['suspicious']:
                    indicators.append(f"Suspicious camera info: {camera_info['details']}")
                    confidence += 0.25
                
                # Check for AI software signatures
                software_check = self._check_software_signatures(exif_data)
                analysis_details['software_check'] = software_check
                
                if software_check['detected']:
                    indicators.append(f"AI software detected: {software_check['tool']}")
                    confidence += 0.5
                
                # Check for suspicious patterns in metadata
                pattern_check = self._check_suspicious_patterns(exif_data)
                analysis_details['pattern_check'] = pattern_check
                
                if pattern_check['found']:
                    indicators.extend([f"Suspicious pattern: {p}" for p in pattern_check['patterns']])
                    confidence += 0.3 * len(pattern_check['patterns'])
                
                # Check for missing essential EXIF tags
                missing_tags = self._check_missing_exif_tags(exif_data)
                analysis_details['missing_tags'] = missing_tags
                
                if missing_tags:
                    indicators.append(f"Missing common EXIF tags: {', '.join(missing_tags)}")
                    confidence += 0.1
                
                # Check creation date consistency
                date_check = self._check_date_consistency(exif_data, file_path)
                analysis_details['date_check'] = date_check
                
                if date_check['suspicious']:
                    indicators.append(f"Date inconsistency: {date_check['details']}")
                    confidence += 0.2
                
        except ImportError:
            warnings.append("PIL/Pillow not installed - limited metadata analysis")
            confidence = 0.0
        except Exception as e:
            warnings.append(f"Error reading image metadata: {str(e)}")
            logger.error(f"Image metadata error: {e}")
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)

        print(f"Image confidence: {confidence}, indicators: {indicators}, warnings: {warnings}, raw_metadata: {raw_metadata}, analysis_details: {analysis_details}")
        
        return MetadataResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            warnings=warnings,
            raw_metadata=raw_metadata,
            analysis_details=analysis_details
        )
    
    def _analyze_video(self, file_path: str) -> MetadataResult:
        """Analyze video metadata"""
        indicators = []
        warnings = []
        raw_metadata = {}
        analysis_details = {}
        confidence = 0.0
        
        try:
            # Try using ffmpeg-python or subprocess to get metadata
            metadata = self._extract_video_metadata_ffprobe(file_path)
            raw_metadata = metadata
            
            if not metadata:
                warnings.append("Could not extract video metadata")
                return MetadataResult(
                    is_suspicious=False,
                    confidence_score=0.0,
                    warnings=warnings
                )
            
            # Check for encoding software
            if 'format' in metadata:
                format_info = metadata['format']
                
                # Check tags
                tags = format_info.get('tags', {})
                
                # Check for AI software in encoder
                encoder = tags.get('encoder', '').lower()
                software_check = self._check_software_in_text(encoder)
                analysis_details['encoder_check'] = software_check
                
                if software_check['detected']:
                    indicators.append(f"AI encoder detected: {software_check['tool']}")
                    confidence += 0.4
                
                # Check for suspicious patterns in all tags
                all_tag_text = ' '.join(str(v) for v in tags.values()).lower()
                pattern_check = self._check_suspicious_patterns({'text': all_tag_text})
                analysis_details['tag_pattern_check'] = pattern_check
                
                if pattern_check['found']:
                    indicators.extend([f"Suspicious tag pattern: {p}" for p in pattern_check['patterns']])
                    confidence += 0.2 * len(pattern_check['patterns'])
                
                # Check for unusual encoding parameters
                encoding_check = self._check_encoding_parameters(metadata)
                analysis_details['encoding_check'] = encoding_check
                
                if encoding_check['suspicious']:
                    indicators.append(f"Unusual encoding: {encoding_check['details']}")
                    confidence += 0.15
                
                # Check creation time
                creation_time = tags.get('creation_time', '')
                if not creation_time:
                    indicators.append("No creation time metadata found")
                    confidence += 0.1
                
            # Check streams for inconsistencies
            if 'streams' in metadata:
                for stream in metadata['streams']:
                    codec = stream.get('codec_name', '')
                    codec_tag = stream.get('codec_tag_string', '')
                    
                    # Check for unusual codecs
                    if codec in ['libvpx-vp9', 'libaom-av1'] and 'webm' not in file_path.lower():
                        indicators.append(f"Unusual codec for container: {codec}")
                        confidence += 0.1
                    
                    # Check for screen recording indicators
                    if 'screen' in codec_tag.lower() or 'capture' in codec_tag.lower():
                        indicators.append("Screen capture codec detected")
                        confidence += 0.2
                
                analysis_details['stream_analysis'] = {
                    'codec_info': [s.get('codec_name', 'unknown') for s in metadata['streams']]
                }
                
        except Exception as e:
            warnings.append(f"Error analyzing video metadata: {str(e)}")
            logger.error(f"Video metadata error: {e}")
        
        confidence = min(confidence, 1.0)
        print(f"Video confidence: {confidence}, indicators: {indicators}, warnings: {warnings}, raw_metadata: {raw_metadata}, analysis_details: {analysis_details}")
        
        return MetadataResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            warnings=warnings,
            raw_metadata=raw_metadata,
            analysis_details=analysis_details
        )
    
    def _analyze_audio(self, file_path: str) -> MetadataResult:
        """Analyze audio metadata"""
        indicators = []
        warnings = []
        raw_metadata = {}
        analysis_details = {}
        confidence = 0.0
        
        try:
            # Try to extract audio metadata
            metadata = self._extract_audio_metadata(file_path)
            raw_metadata = metadata
            
            if not metadata:
                # Fallback to basic file analysis
                indicators.append("Limited audio metadata available")
                confidence += 0.1
            else:
                # Check for AI audio generation signatures
                tags = metadata.get('tags', {})
                
                # Check all tag values for AI signatures
                all_tag_text = ' '.join(str(v) for v in tags.values()).lower()
                software_check = self._check_software_in_text(all_tag_text)
                analysis_details['software_check'] = software_check
                
                if software_check['detected']:
                    indicators.append(f"AI audio tool detected: {software_check['tool']}")
                    confidence += 0.5
                
                # Check for TTS indicators
                pattern_check = self._check_suspicious_patterns({'text': all_tag_text})
                analysis_details['pattern_check'] = pattern_check
                
                if pattern_check['found']:
                    indicators.extend([f"Suspicious pattern: {p}" for p in pattern_check['patterns']])
                    confidence += 0.25 * len(pattern_check['patterns'])
                
                # Check for missing common audio metadata
                if not tags.get('artist') and not tags.get('album'):
                    indicators.append("Missing artist/album metadata")
                    confidence += 0.1
                
        except Exception as e:
            warnings.append(f"Error analyzing audio metadata: {str(e)}")
            logger.error(f"Audio metadata error: {e}")
        
        confidence = min(confidence, 1.0)

        print(f"Audio confidence: {confidence}, indicators: {indicators}, warnings: {warnings}, raw_metadata: {raw_metadata}, analysis_details: {analysis_details}")
        
        return MetadataResult(
            is_suspicious=confidence > 0.3,
            confidence_score=confidence,
            indicators=indicators,
            warnings=warnings,
            raw_metadata=raw_metadata,
            analysis_details=analysis_details
        )
    
    def _check_camera_info(self, exif_data: Dict) -> Dict:
        """Check if camera information is legitimate"""
        result = {
            'missing': False,
            'suspicious': False,
            'details': '',
            'camera': None
        }
        
        make = exif_data.get('Make', '').lower() if exif_data.get('Make') else ''
        model = exif_data.get('Model', '') if exif_data.get('Model') else ''
        
        if not make and not model:
            result['missing'] = True
            result['details'] = 'No camera make/model found'
            return result
        
        result['camera'] = f"{make} {model}".strip()
        
        # Check if make is in legitimate cameras
        if make and not any(cam in make for cam in self.legitimate_cameras):
            result['suspicious'] = True
            result['details'] = f"Unknown camera manufacturer: {make}"
        
        return result
    
    def _check_software_signatures(self, exif_data: Dict) -> Dict:
        """Check for AI software signatures in EXIF data"""
        result = {
            'detected': False,
            'tool': None,
            'matched_signature': None
        }
        
        # Check common EXIF fields for software signatures
        fields_to_check = [
            exif_data.get('Software', ''),
            exif_data.get('ProcessingSoftware', ''),
            exif_data.get('Artist', ''),
            exif_data.get('ImageDescription', ''),
            exif_data.get('UserComment', ''),
        ]
        
        combined_text = ' '.join(str(f) for f in fields_to_check if f).lower()
        
        for tool_name, signatures in self.ai_signatures.items():
            for sig in signatures:
                if sig.lower() in combined_text:
                    result['detected'] = True
                    result['tool'] = tool_name
                    result['matched_signature'] = sig
                    return result
        
        return result
    
    def _check_software_in_text(self, text: str) -> Dict:
        """Check for AI software signatures in any text"""
        result = {
            'detected': False,
            'tool': None,
            'matched_signature': None
        }
        
        text_lower = text.lower()
        
        for tool_name, signatures in self.ai_signatures.items():
            for sig in signatures:
                if sig.lower() in text_lower:
                    result['detected'] = True
                    result['tool'] = tool_name
                    result['matched_signature'] = sig
                    return result
        
        return result
    
    def _check_suspicious_patterns(self, data: Dict) -> Dict:
        """Check for suspicious patterns in metadata"""
        result = {
            'found': False,
            'patterns': []
        }
        
        # Combine all values into searchable text
        all_text = ' '.join(str(v) for v in data.values() if v).lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                result['found'] = True
                result['patterns'].append(pattern)
        
        return result
    
    def _check_missing_exif_tags(self, exif_data: Dict) -> List[str]:
        """Check for commonly expected EXIF tags"""
        expected_tags = [
            'DateTimeOriginal',
            'ExifIFDPointer',
            'ExifImageWidth',
            'ExifImageHeight',
        ]
        
        missing = []
        for tag in expected_tags:
            if tag not in exif_data or not exif_data[tag]:
                missing.append(tag)
        
        return missing
    
    def _check_date_consistency(self, exif_data: Dict, file_path: str) -> Dict:
        """Check if dates in metadata are consistent"""
        result = {
            'suspicious': False,
            'details': ''
        }
        
        # Get EXIF dates
        date_original = exif_data.get('DateTimeOriginal', '')
        date_digitized = exif_data.get('DateTimeDigitized', '')
        # date_modified = exif_data.get('DateTime', '')  # Available for future checks
        
        # Get file system dates (available for future consistency checks)
        # try:
        #     stat = os.stat(file_path)
        #     file_mtime = datetime.fromtimestamp(stat.st_mtime)
        # except Exception:
        #     file_mtime = None
        
        # Check if dates are suspiciously identical (common in AI-generated)
        if date_original and date_digitized:
            if date_original == date_digitized:
                result['suspicious'] = True
                result['details'] = 'Original and digitized dates are identical'
        
        return result
    
    def _check_encoding_parameters(self, metadata: Dict) -> Dict:
        """Check for unusual encoding parameters"""
        result = {
            'suspicious': False,
            'details': ''
        }
        
        format_info = metadata.get('format', {})
        
        # Check for unusual bitrates
        bitrate = int(format_info.get('bit_rate', 0))
        if bitrate > 0:
            # Very low or very high bitrates can be suspicious
            duration = float(format_info.get('duration', 0))
            if duration > 0:
                # Check for unusual bitrate patterns
                pass  # Add specific checks if needed
        
        # Check for unusual format combinations
        format_name = format_info.get('format_name', '')
        if 'webm' in format_name or 'matroska' in format_name:
            # These are common for AI-generated videos
            pass  # Could add scoring
        
        return result
    
    def _extract_video_metadata_ffprobe(self, file_path: str) -> Dict:
        """Extract video metadata using ffprobe"""
        try:
            import subprocess
            import json as json_module
            
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json_module.loads(result.stdout)
            else:
                logger.warning(f"ffprobe failed: {result.stderr}")
                return {}
                
        except FileNotFoundError:
            logger.warning("ffprobe not found - video metadata extraction limited")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out")
            return {}
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return {}
    
    def _extract_audio_metadata(self, file_path: str) -> Dict:
        """Extract audio metadata"""
        metadata = {}
        
        try:
            # Try using mutagen for audio metadata
            try:
                from mutagen import File
                audio = File(file_path)
                if audio:
                    metadata['tags'] = dict(audio.tags) if hasattr(audio, 'tags') and audio.tags else {}
                    metadata['info'] = {
                        'length': audio.info.length if hasattr(audio, 'info') else 0,
                        'bitrate': audio.info.bitrate if hasattr(audio, 'info') else 0,
                    }
            except ImportError:
                pass
            
            # Fallback to ffprobe
            if not metadata:
                metadata = self._extract_video_metadata_ffprobe(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting audio metadata: {e}")
        
        return metadata


def analyze_metadata(file_path: str) -> MetadataResult:
    """
    Convenience function to analyze a file's metadata.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        MetadataResult with analysis details
    """
    analyzer = MetadataAnalyzer()
    return analyzer.analyze_file(file_path)


def get_metadata_summary(result: MetadataResult) -> str:
    """
    Generate a human-readable summary of metadata analysis.
    
    Args:
        result: MetadataResult from analysis
        
    Returns:
        Formatted summary string
    """
    if result.is_suspicious:
        status = "⚠️ SUSPICIOUS"
        confidence_pct = result.confidence_score * 100
        summary = f"{status} - {confidence_pct:.1f}% confidence of AI generation\n"
    else:
        summary = "✅ No suspicious metadata detected\n"
    
    if result.indicators:
        summary += "\nIndicators found:\n"
        for indicator in result.indicators:
            summary += f"  • {indicator}\n"
    
    if result.warnings:
        summary += "\nWarnings:\n"
        for warning in result.warnings:
            summary += f"  ⚠️ {warning}\n"
    
    return summary


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python metadata_analyzer.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = analyze_metadata(file_path)
    
    print(get_metadata_summary(result))
    print("\nDetailed Analysis:")
    print(json.dumps(result.analysis_details, indent=2, default=str))
