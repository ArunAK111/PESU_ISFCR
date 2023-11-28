import torch
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

from torchvision import models, transforms

input_video_folder = r"C:\Users\arunk\Desktop\1-1004"
output_npy_folder = r"C:\Users\arunk\Desktop\New folder\i3d_features"
segment_length = 200

if not os.path.exists(output_npy_folder):
    os.makedirs(output_npy_folder)

i3d_model = models.video.i3d.InceptionI3d(num_classes=400, in_channels=3)
i3d_model.eval()

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 1:
        return None

    frames = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_transform(frame)
            frames.append(frame)

    cap.release()
    return torch.stack(frames).to(device)

def extract_i3d_features(video_frames):
    with torch.no_grad():
        features = i3d_model.extract_features(video_frames)
    return features.cpu().numpy()

video_files = glob.glob(os.path.join(input_video_folder, '*.mp4'))
for video_file in tqdm(video_files, desc="Processing videos"):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_frames = preprocess_video(video_file)

    if len(video_frames) > 0:
        features = extract_i3d_features(video_frames)

        for i, feature in enumerate(features):
            npy_path = os.path.join(output_npy_folder, f"{video_name}_feature_{i}.npy")
            np.save(npy_path, feature)
