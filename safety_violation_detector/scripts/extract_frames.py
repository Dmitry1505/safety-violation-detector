import cv2
import os
import argparse

def extract_frames(video_path, outdir, fps=1):
    os.makedirs(outdir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    video_fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(int(video_fps / fps), 1)
    count, saved = 0, 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(outdir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    print(f"Extracted {saved} frames to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()
    extract_frames(args.video, args.outdir, args.fps)
