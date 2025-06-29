import streamlit as st
import requests

st.title("🎬 Video Summarization App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    with st.spinner("Uploading and summarizing..."):
        files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
        res = requests.post("http://localhost:5000/upload", files=files)

    if res.status_code == 200:
        data = res.json()
        st.success("✅ Summary generated successfully!")

        # Display original video
        st.subheader("📤 Uploaded Video")
        st.video(data["video_path"])

        # Display summary video
        st.subheader("🎞️ Summary Video")
        st.video(data["summary_path"])

        # Download button
        with open(data["summary_path"], 'rb') as f:
            st.download_button("⬇ Download Summary", f, file_name="summary_video.mp4")
    else:
        st.error("❌ Upload or summarization failed.")
