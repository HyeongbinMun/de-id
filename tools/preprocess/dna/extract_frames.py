import os
import sys
import cv2
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def extract_frames(video_path, frame_dir, fps):
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(video_fps / fps) == 0:
            frame_path = f"{frame_dir}/{video_name}_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_count += 1

    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract frames from videos at a specified fps to fit the YOLO dataset directory structure.")
    parser.add_argument("--video_dir", type=str, default="/dataset/dna/videos", help="Directory path of the original videos")
    parser.add_argument("--frame_dir", type=str, default="/dataset/dna/frames", help="Target directory to save the frame images")
    parser.add_argument("--fps", type=int, default=1, help="Number of frames per second to extract")

    options = parser.parse_known_args()[0]
    video_dir = options.video_dir
    frame_dir = options.frame_dir
    fps = options.fps

    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    video_names = os.listdir(video_dir)
    progress_bar = tqdm(enumerate(video_names), total=len(video_names))
    for i, video_name in progress_bar:
        progress_bar.set_description(f"Processing({video_name})")
        video_path = os.path.join(video_dir, video_name)
        if os.path.isfile(video_path) and video_name.lower().endswith(('.mp4', '.avi', '.mov')):
            extract_frames(video_path, frame_dir, fps)
