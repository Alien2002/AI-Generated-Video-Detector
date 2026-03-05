"""
Example: Using GenConViT ONNX for Video Deepfake Detection

This shows how to integrate GenConViT ONNX with your existing video processing pipeline.
"""
import cv2
import numpy as np
from models.genconvit_onnx import load_genconvit_onnx

# Load GenConViT ONNX models (do this once at startup)
genconvit = load_genconvit_onnx(
    weights_dir='checkpoints/genconvit_weights',
    use_ed=True,   # Use ED (Autoencoder) model
    use_vae=True   # Use VAE model (ensemble gives best results)
)


def extract_faces_from_frame(frame, face_detector=None):
    """
    Extract faces from a video frame.
    Use your existing MTCNN or face_recognition here.
    """
    # Example with OpenCV Haar Cascade (replace with your MTCNN)
    if face_detector is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        face_cascade = face_detector
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_images = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_images.append(face_rgb)
    
    return face_images


def predict_video_genconvit(video_path, num_frames=15):
    """
    Predict if a video is fake using GenConViT ONNX.
    
    This can replace or complement your existing deepfakes_video_predict()
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample evenly spaced frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    all_faces = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in indices:
            faces = extract_faces_from_frame(frame)
            all_faces.extend(faces)
        
        frame_idx += 1
    
    cap.release()
    
    # Run GenConViT prediction on all detected faces
    result = genconvit.predict_video_frames(all_faces)
    
    return result


# Example usage
if __name__ == "__main__":
    # Test on a video
    video_path = "videos/test_video.mp4"
    
    result = predict_video_genconvit(video_path, num_frames=15)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Frames analyzed: {result['frame_count']}")
    print(f"Fake frames: {result['fake_frames']}")
    print(f"Real frames: {result['real_frames']}")


# Integration with your existing inference.py:
"""
In your inference.py, add:

from models.genconvit_onnx import load_genconvit_onnx

# Load at module level
genconvit = load_genconvit_onnx('checkpoints/genconvit_weights')

def deepfakes_video_predict_genconvit(input_video, num_frames=15):
    # Use your existing preprocess_video to get frames
    video_frames = preprocess_video(input_video, num_frames)
    
    # Extract faces (use your existing face detection)
    faces = []
    for frame in video_frames:
        # Your face detection here
        face = extract_face(frame)  # Returns RGB face or None
        if face is not None:
            faces.append(face)
    
    # Run GenConViT
    result = genconvit.predict_video_frames(faces)
    
    text = f"The video is {result['prediction']}."
    return text, result['confidence']
"""
