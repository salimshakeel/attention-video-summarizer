import streamlit as st
import os
import sys
from model.model_loader import summarize_video
from datetime import datetime
import shutil

# === Directories ===
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Streamlit App ===
st.set_page_config(page_title="Video Summarizer", layout="centered")
st.title("🎬 AI Video Summarizer (PGL-SUM Based)")

st.markdown("""
Upload a video, and our model will generate a summarized version automatically using deep learning.
""")

# === File Upload ===
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"video_{timestamp}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success("✅ Video uploaded successfully!")
    st.video(video_path)

    # Start summarization
    if st.button("📽️ Generate Summary"):
        with st.spinner("Summarizing video... This may take a moment."):
            summary_path = os.path.join(OUTPUT_FOLDER, f"summary_{timestamp}.mp4")
            try:
                summarize_video(video_path, summary_path)
                st.success("✅ Summary video generated!")
                st.video(summary_path)

                # Download option
                with open(summary_path, "rb") as video_file:
                    st.download_button(
                        label="📥 Download Summary",
                        data=video_file,
                        file_name=f"summary_{timestamp}.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
