"""
Deepfake Detection - Inference Pipeline
========================================
Load model dan predict video baru dengan preprocessing yang sama seperti training.

Usage:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model('deepfake_model_best.pth', device)
    preprocessor = VideoPreprocessor(device=device)
    result = predict_video('test.mp4', model, preprocessor)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import glob
from facenet_pytorch import MTCNN


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        f = self.features(x)
        return self.pool(f).flatten(1)


class EffB0_BiGRU(nn.Module):
    def __init__(self, hidden=256, bidirectional=True):
        super().__init__()
        self.enc = FrameEncoder()
        self.gru = nn.GRU(1280, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(out_dim, 1)
    
    def forward(self, x):
        B, T = x.shape[:2]
        x = x.view(B*T, 3, 224, 224)
        f = self.enc(x).view(B, T, -1)
        _, h = self.gru(f)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.gru.bidirectional else h[-1]
        return self.fc(self.drop(h)).squeeze(1)


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(checkpoint_path, device='cuda'):
    """Load model dari checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EffB0_BiGRU(hidden=256, bidirectional=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    
    print(f" Model loaded: {checkpoint_path}")
    if 'metrics' in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")
    
    return model, checkpoint


# ============================================================================
# PREPROCESSING
# ============================================================================

class VideoPreprocessor:
    """Preprocessing pipeline - HARUS sama dengan training!"""
    
    def __init__(self, num_frames=8, frame_size=224, device='cuda'):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.device = device
        self.face_detector = MTCNN(
            image_size=frame_size, margin=0, keep_all=False, device=device
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_mid_clip_frames(self, video_path, seconds=2.0):
        """Extract 8 frames dari mid-clip 2 detik"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid video: fps={fps}, frames={total_frames}")
        
        target_frames = int(seconds * fps)
        mid_frame = total_frames // 2
        start = max(0, mid_frame - target_frames // 2)
        end = min(total_frames, start + target_frames)
        
        indices = np.linspace(start, end - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        cap.release()
        
        # Pad jika kurang
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        
        return frames[:self.num_frames]
    
    def detect_and_crop_face(self, image):
        """Detect & crop face dengan MTCNN"""
        face = self.face_detector(image)
        
        if face is None:
            # Fallback: center crop
            w, h = image.size
            size = min(w, h)
            left, top = (w - size) // 2, (h - size) // 2
            cropped = image.crop((left, top, left + size, top + size))
            return cropped.resize((self.frame_size, self.frame_size), Image.BILINEAR)
        
        # Convert tensor to PIL
        face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(face_np)
    
    def preprocess_video(self, video_path):
        """Full pipeline: video → preprocessed tensor"""
        frames = self.extract_mid_clip_frames(video_path)
        faces = [self.detect_and_crop_face(f) for f in frames]
        tensors = [self.transform(f) for f in faces]
        return torch.stack(tensors, dim=0).unsqueeze(0)


# ============================================================================
# PREDICTION
# ============================================================================

def predict_video(video_path, model, preprocessor, threshold=0.85, device='cuda'):
    """
    Predict video: REAL atau FAKE
    
    Returns:
        dict dengan keys: prediction, probability, confidence, raw_logit
    """
    try:
        video_tensor = preprocessor.preprocess_video(video_path).to(device)
    except Exception as e:
        return {'error': str(e), 'prediction': None, 'probability': None}
    
    model.eval()
    with torch.no_grad():
        logit = model(video_tensor)
        prob = torch.sigmoid(logit).item()
    
    pred = 'REAL' if prob >= threshold else 'FAKE'
    conf = prob if pred == 'REAL' else (1 - prob)
    
    return {
        'prediction': pred,
        'probability': prob,
        'confidence': conf,
        'raw_logit': logit.item(),
        'threshold_used': threshold
    }


def batch_inference(video_folder, model, preprocessor, threshold=0.85, device='cuda'):
    """Predict semua video dalam folder"""
    exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    paths = []
    for ext in exts:
        paths.extend(glob.glob(str(Path(video_folder) / ext)))
    
    print(f"Found {len(paths)} videos")
    
    results = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {Path(path).name}", end=' ')
        result = predict_video(path, model, preprocessor, threshold, device)
        result['video_path'] = path
        result['video_name'] = Path(path).name
        results.append(result)
        print(f"→ {result['prediction']} ({result['probability']:.3f})")
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load
    model, _ = load_model('deepfake_model_best.pth', device=device)
    preprocessor = VideoPreprocessor(device=device)
    
    # Predict single video
    result = predict_video('test.mp4', model, preprocessor)
    print(f"\n{result['prediction']} - Confidence: {result['confidence']:.1%}")
    
    # Batch predict
    # results = batch_inference('videos/', model, preprocessor)
