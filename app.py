"""
Deepfake Detection - Streamlit UI
==================================
Web interface for detecting deepfake videos using EfficientNet-B0 + BiGRU model.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import tempfile
import time
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from deepfake_inference_v2 import load_model, VideoPreprocessorV2, predict_video


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def get_model():
    """Load model once and cache it"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(__file__).parent / 'models' / 'deepfake_model_best.pth'
    model, checkpoint = load_model(str(model_path), device)
    # Use V2 preprocessor that matches training (Haar Cascade + 20% margin)
    preprocessor = VideoPreprocessorV2(num_frames=8, frame_size=224, margin=0.20)
    return model, preprocessor, device


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Deepfake Detection",
        page_icon="🔍",
        layout="wide"
    )

    # Header
    st.title("🔍 Deepfake Detection System")
    st.caption("EfficientNet-B0 + BiGRU | Trained on DFD Dataset")
    st.markdown("---")

    # Load model
    with st.spinner("Loading model..."):
        model, preprocessor, device = get_model()

    # Sidebar - Device info
    st.sidebar.header("System Info")
    device_str = "🟢 CUDA (GPU)" if device.type == 'cuda' else "🟡 CPU"
    st.sidebar.success(f"Device: {device_str}")
    st.sidebar.info(f"Model: EfficientNet-B0 + BiGRU")
    st.sidebar.info(f"Frames: 8 (mid-clip 2s)")
    st.sidebar.info(f"Threshold: 0.85")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )

        if uploaded_file is not None:
            # Show video preview
            st.video(uploaded_file)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Analyze button
            analyze_btn = st.button("🔎 Analyze Video", type="primary", use_container_width=True)

            if analyze_btn:
                with col2:
                    st.subheader("📊 Analysis Results")

                    # Progress container
                    progress_container = st.container()

                    with progress_container:
                        # Step 1: Extract frames
                        with st.status("Processing video...", expanded=True) as status:
                            st.write("1️⃣ Extracting frames from video...")
                            start_time = time.time()
                            frames_bgr = preprocessor.extract_mid_clip_frames(video_path)
                            st.write(f"   ✅ Extracted {len(frames_bgr)} frames")

                            # Step 2: Detect faces (Haar Cascade + anchor bbox)
                            st.write("2️⃣ Detecting and cropping faces (Haar Cascade)...")
                            faces = preprocessor.get_face_images(video_path)
                            st.write(f"   ✅ Processed {len(faces)} face crops")

                            # Step 3: Run inference
                            st.write("3️⃣ Running model inference...")
                            result = predict_video(video_path, model, preprocessor, threshold=0.85, device=device)
                            elapsed_time = time.time() - start_time

                            if result.get('error'):
                                status.update(label="Error!", state="error")
                                st.error(f"Error: {result['error']}")
                            else:
                                status.update(label="Analysis complete!", state="complete")

                    # Display results if successful
                    if not result.get('error'):
                        st.markdown("---")

                        # Face frames grid
                        st.subheader("👤 Extracted Faces")
                        face_cols = st.columns(8)
                        for i, face in enumerate(faces):
                            with face_cols[i]:
                                st.image(face, caption=f"Frame {i+1}", use_container_width=True)

                        st.markdown("---")

                        # Prediction result
                        st.subheader("🎯 Prediction")

                        pred = result['prediction']
                        conf = result['confidence']
                        prob = result['probability']

                        # Large result display
                        result_col1, result_col2 = st.columns([1, 2])

                        with result_col1:
                            if pred == 'REAL':
                                st.success(f"## ✅ {pred}")
                            else:
                                st.error(f"## ⚠️ {pred}")

                        with result_col2:
                            st.metric(
                                label="Confidence",
                                value=f"{conf:.1%}",
                                delta=None
                            )
                            st.progress(conf)

                        # Detailed metrics
                        st.markdown("---")
                        with st.expander("📈 Detailed Metrics", expanded=True):
                            metric_cols = st.columns(4)

                            with metric_cols[0]:
                                st.metric("Probability", f"{prob:.4f}")

                            with metric_cols[1]:
                                st.metric("Raw Logit", f"{result['raw_logit']:.4f}")

                            with metric_cols[2]:
                                st.metric("Threshold", f"{result['threshold_used']}")

                            with metric_cols[3]:
                                st.metric("Processing Time", f"{elapsed_time:.2f}s")

                        # Interpretation
                        st.markdown("---")
                        st.subheader("📝 Interpretation")
                        if pred == 'REAL':
                            st.info(
                                f"The model predicts this video is **REAL** with {conf:.1%} confidence. "
                                f"The probability score ({prob:.4f}) is above the threshold, "
                                f"indicating authentic video content."
                            )
                        else:
                            st.warning(
                                f"The model predicts this video is **FAKE** with {conf:.1%} confidence. "
                                f"The probability score ({prob:.4f}) is below the threshold, "
                                f"indicating potential deepfake manipulation."
                            )

                # Cleanup temp file
                try:
                    os.unlink(video_path)
                except:
                    pass

    # Footer
    st.markdown("---")
    st.caption("Deepfake Detection System | EfficientNet-B0 + BiGRU | DFD Dataset")


if __name__ == "__main__":
    main()
