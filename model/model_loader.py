import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from layers.summarizer import PGL_SUM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Feature Extractor ---
def get_feature_extractor():
    model = models.googlenet(pretrained=True).to(device).eval()
    extractor = torch.nn.Sequential(
        model.conv1,
        model.maxpool1,
        model.conv2,
        model.conv3,
        model.maxpool2,
        model.inception3a,
        model.inception3b,
        model.maxpool3,
        model.inception4a,
        model.inception4b,
        model.inception4c,
        model.inception4d,
        model.inception4e,
        model.maxpool4,
        model.inception5a,
        model.inception5b,
        model.avgpool,
        torch.nn.Flatten(),
    )
    return extractor

# --- 2. Frame Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 3. Frame Extraction + Feature Extraction ---
def extract_video_features(video_path, feature_extractor, frame_rate=15):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, picks, original = [], [], []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % round(fps // frame_rate)) == 0:
            original.append(frame)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = feature_extractor(input_tensor)
                frames.append(feature.squeeze(0).cpu().numpy())
                picks.append(count)
        count += 1
    cap.release()
    return frames, picks, original

# --- 4. Frame Selection ---
def select_frames(scores, picks, threshold=0.4):
    return [picks[i] for i, score in enumerate(scores) if score >= threshold]

def get_selected_frames(video_path, selected_indices):
    cap = cv2.VideoCapture(video_path)
    selected_set = set(selected_indices)
    selected_frames, frame_id = {}, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in selected_set:
            selected_frames[frame_id] = frame
        frame_id += 1
    cap.release()
    return selected_frames

# --- 5. Write Summary Video ---
def write_summary_video(output_path, selected_frames, fps=15):
    if not selected_frames:
        print("⚠️ No frames selected.")
        return

    height, width, _ = list(selected_frames.values())[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for fid in sorted(selected_frames.keys()):
        out.write(selected_frames[fid])
    out.release()
    print(f"✅ Summary saved: {output_path}")

# --- 🔥 Main Pipeline ---
def summarize_video(video_path, output_path="summary_output.mp4"):
    print("[INFO] Starting summarization...")
    extractor = get_feature_extractor()

    frames, picks, original = extract_video_features(video_path, extractor)
    print(f"[INFO] {len(frames)} frame features extracted.")

    frames_tensor = torch.tensor(frames, dtype=torch.float32).to(device)

    model = PGL_SUM(
        input_size=1024,
        output_size=1024,
        num_segments=4,
        heads=8,
        fusion="add",
        pos_enc="absolute"
    )
    model.load_state_dict(torch.load("Model/epoch-199.pkl", map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        scores, _ = model(frames_tensor)
        scores = scores.squeeze(0).cpu().numpy().tolist()

    selected_indices = select_frames(scores, picks, threshold=0.4)
    print(f"[INFO] Selected {len(selected_indices)} keyframes.")

    selected_frames = get_selected_frames(video_path, selected_indices)
    write_summary_video(output_path, selected_frames)

    return output_path
