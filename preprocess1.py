import torch
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

# Define parameters
input_video_folder = r"C:\Users\arunk\Desktop\1-1004"
output_npy_folder = r"C:\Users\arunk\Desktop\New folder\i3d"
segment_length = 200

# Create output folder if not exists
if not os.path.exists(output_npy_folder):
    os.makedirs(output_npy_folder)

# Function to preprocess a video
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 1:
        return None

    frames = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_transform = torch.nn.functional.interpolate

    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            crop_size = 224
            crops = [
                frame_tensor[:, :, :crop_size, :crop_size],
                frame_tensor[:, :, -crop_size:, :crop_size],
                frame_tensor[:, :, :crop_size, -crop_size:],
                frame_tensor[:, :, -crop_size:, -crop_size:],
                frame_tensor[:, :, frame_tensor.size(2) // 2 - crop_size // 2:frame_tensor.size(2) // 2 + crop_size // 2,
                                  frame_tensor.size(3) // 2 - crop_size // 2:frame_tensor.size(3) // 2 + crop_size // 2]
            ]

            for crop in crops:
                resized_frame = resize_transform(crop, size=(224, 224), mode='bilinear', align_corners=False)
                frames.append(resized_frame.cpu().numpy())

    cap.release()
    return np.array(frames)

# Process videos in the input folder
video_files = glob.glob(os.path.join(input_video_folder, '*.mp4'))
for video_file in tqdm(video_files, desc="Processing videos"):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_data = preprocess_video(video_file)

    if video_data is not None:
        for i, cropped_frame in enumerate(video_data):
            npy_path = os.path.join(output_npy_folder, f"{video_name}_frame_{i}.npy")
            np.save(npy_path, cropped_frame)
