"""
Deepfake Detection - Inference Pipeline V2
===========================================
Inference yang MATCH dengan preprocessing training di Kaggle.

Perbedaan dari v1:
- Menggunakan Haar Cascade (bukan MTCNN) untuk face detection
- Margin 20% seperti training
- Anchor bbox dengan median untuk stabilitas antar-frame
- Resize dengan cv2.INTER_AREA

Usage:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model('deepfake_model_best.pth', device)
    preprocessor = VideoPreprocessorV2()
    result = predict_video('test.mp4', model, preprocessor, device=device)
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


# ============================================================================
# MODEL ARCHITECTURE (sama dengan training)
# ============================================================================

class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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
        x = x.view(B * T, 3, 224, 224)
        f = self.enc(x).view(B, T, -1)
        _, h = self.gru(f)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.gru.bidirectional else h[-1]
        return self.fc(self.drop(h)).squeeze(1)


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(checkpoint_path, device='cuda'):
    """Load model dari checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EffB0_BiGRU(hidden=256, bidirectional=True)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict (from Kaggle training)
        model.load_state_dict(checkpoint)

    model.eval().to(device)

    print(f"[OK] Model loaded: {checkpoint_path}")
    if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")

    return model, checkpoint


# ============================================================================
# PREPROCESSING V2 - MATCHING TRAINING
# ============================================================================

class VideoPreprocessorV2:
    """
    Preprocessing pipeline yang MATCH dengan training di Kaggle.

    Key differences from v1:
    - Uses Haar Cascade (same as training) instead of MTCNN
    - 20% margin around face bbox
    - Anchor bbox using median across frames for stability
    - cv2.INTER_AREA for resize
    """

    # ImageNet normalization (same as training)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def __init__(self, num_frames=8, frame_size=224, margin=0.20):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.margin = margin

        # Haar Cascade - SAME as training
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Transform - SAME as training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.IMG_MEAN, self.IMG_STD)
        ])

    def extract_mid_clip_frames(self, video_path, seconds=2.0):
        """
        Extract frames dari mid-clip 2 detik - SAME as training.
        Returns list of BGR numpy arrays.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video: fps={fps}, frames={total_frames}")

        # Calculate mid-clip indices - SAME logic as training
        need = int(seconds * fps) if fps > 0 else max(self.num_frames, 8)

        if total_frames <= need:
            # Short video: uniform sampling
            k_eff = min(self.num_frames, total_frames)
            if k_eff <= 1:
                indices = [0]
            else:
                indices = [min(int(round(i * (total_frames - 1) / (k_eff - 1))), total_frames - 1)
                          for i in range(k_eff)]
        else:
            # Mid-clip sampling
            start = max(0, (total_frames - need) // 2)
            end = start + need
            span = list(range(start, end))
            if self.num_frames >= len(span):
                indices = span
            else:
                indices = [span[min(int(round(i * (len(span) - 1) / (self.num_frames - 1))), len(span) - 1)]
                          for i in range(self.num_frames)]

        # Read frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)  # Keep as BGR for Haar cascade

        cap.release()

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def detect_face_bgr(self, img_bgr):
        """Detect face using Haar Cascade - SAME as training"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Return largest face (x, y, w, h)
        return max(faces, key=lambda b: b[2] * b[3])

    def clamp_bbox(self, x0, y0, x1, y1, W, H):
        """Clamp bbox to image boundaries"""
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        return x0, y0, x1, y1

    def compute_anchor_bbox(self, frames_bgr):
        """
        Compute stable face bbox using MEDIAN across frames - SAME as training.
        This provides temporal consistency.
        """
        H, W = None, None
        boxes = []

        for img in frames_bgr:
            if img is None:
                continue
            if H is None:
                H, W = img.shape[:2]

            det = self.detect_face_bgr(img)
            if det is not None:
                x, y, w, h = det
                cx, cy = x + w / 2, y + h / 2
                s = max(w, h)
                boxes.append((cx, cy, s))

        if not boxes:
            # Fallback: center square crop
            s = min(H, W)
            x0, y0 = (W - s) // 2, (H - s) // 2
            return self.clamp_bbox(x0, y0, x0 + s, y0 + s, W, H)

        # Use MEDIAN for stability (same as training)
        boxes = np.array(boxes)
        cx, cy, s = np.median(boxes, axis=0)

        # Add margin (20% like training)
        s = s * (1.0 + self.margin * 2)

        x0 = int(round(cx - s / 2))
        y0 = int(round(cy - s / 2))
        x1 = int(round(cx + s / 2))
        y1 = int(round(cy + s / 2))

        return self.clamp_bbox(x0, y0, x1, y1, W, H)

    def crop_and_resize_faces(self, frames_bgr):
        """
        Crop faces using anchor bbox and resize to 224x224.
        Returns list of RGB PIL Images.
        """
        x0, y0, x1, y1 = self.compute_anchor_bbox(frames_bgr)

        faces = []
        last_face = None

        for frame in frames_bgr:
            if frame is None:
                if last_face is not None:
                    faces.append(last_face)
                continue

            # Crop with anchor bbox
            crop = frame[y0:y1, x0:x1]

            # Resize with INTER_AREA (same as training)
            crop = cv2.resize(crop, (self.frame_size, self.frame_size),
                            interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB for PIL
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(crop_rgb)

            faces.append(face_pil)
            last_face = face_pil

        # Pad if needed
        while len(faces) < self.num_frames:
            faces.append(last_face if last_face else Image.new('RGB', (224, 224)))

        return faces[:self.num_frames]

    def preprocess_video(self, video_path):
        """
        Full pipeline: video → preprocessed tensor.
        MATCHES training preprocessing exactly.
        """
        # 1. Extract mid-clip frames (BGR)
        frames_bgr = self.extract_mid_clip_frames(video_path)

        # 2. Crop faces with anchor bbox + margin (RGB PIL)
        faces = self.crop_and_resize_faces(frames_bgr)

        # 3. Transform to tensor
        tensors = [self.transform(f) for f in faces]

        # 4. Stack: [T, 3, 224, 224] → [1, T, 3, 224, 224]
        return torch.stack(tensors, dim=0).unsqueeze(0)

    def get_face_images(self, video_path):
        """
        Get face crops as PIL Images (for visualization in UI).
        """
        frames_bgr = self.extract_mid_clip_frames(video_path)
        return self.crop_and_resize_faces(frames_bgr)


# ============================================================================
# PREDICTION
# ============================================================================

def predict_video(video_path, model, preprocessor, threshold=0.85, device='cuda'):
    """
    Predict video: REAL atau FAKE

    Label convention (same as training):
    - Label 1 = REAL → high probability = REAL
    - Label 0 = FAKE → low probability = FAKE

    Args:
        video_path: Path to video file
        model: Loaded model
        preprocessor: VideoPreprocessorV2 instance
        threshold: Classification threshold (default 0.5)
        device: torch device

    Returns:
        dict with keys: prediction, probability, confidence, raw_logit, threshold_used
    """
    try:
        video_tensor = preprocessor.preprocess_video(video_path).to(device)
    except Exception as e:
        return {'error': str(e), 'prediction': None, 'probability': None}

    model.eval()
    with torch.no_grad():
        logit = model(video_tensor)
        prob = torch.sigmoid(logit).item()

    # Label convention: prob >= threshold → REAL, prob < threshold → FAKE
    pred = 'REAL' if prob >= threshold else 'FAKE'

    # Confidence: how sure are we about the prediction
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
        print(f"-> {result['prediction']} ({result['probability']:.3f})")

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'deepfake_model_best.pth'
    model, _ = load_model(str(model_path), device=device)

    # Create preprocessor (matching training)
    preprocessor = VideoPreprocessorV2(num_frames=8, frame_size=224, margin=0.20)

    # Predict single video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'test.mp4'

    print(f"\nProcessing: {video_path}")
    result = predict_video(video_path, model, preprocessor, device=device)

    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        print(f"\n{'='*40}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Raw logit: {result['raw_logit']:.4f}")
        print(f"Threshold: {result['threshold_used']}")
        print(f"{'='*40}")
