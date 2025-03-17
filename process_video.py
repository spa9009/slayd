import os
import subprocess
import time

# File paths
VIDEO_PATH = '/Users/harshporwal/Documents/Dev/experiments/input_1.mp4'
FRAMES_DIR = 'extracted_frames'
CLEAN_FRAMES_DIR = 'clean_frames_fashion'
BEST_FRAMES_DIR = 'best_fashion_frames'

# Create directories if they don't exist
for directory in [FRAMES_DIR, CLEAN_FRAMES_DIR, BEST_FRAMES_DIR]:
    os.makedirs(directory, exist_ok=True)
    # Clean existing files
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))

def extract_frames():
    """Extract frames from video using cv2"""
    import cv2
    
    print(f"\nExtracting frames from {VIDEO_PATH}...")
    video = cv2.VideoCapture(VIDEO_PATH)
    
    if not video.isOpened():
        raise Exception("Could not open video file")
    
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        frame_path = os.path.join(FRAMES_DIR, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count} frames...")
    
    video.release()
    print(f"Extracted {frame_count} frames to {FRAMES_DIR}")

def main():
    try:
        # Step 1: Extract frames
        print("\nStep 1: Extracting frames from video...")
        extract_frames()
        
        # Step 2: Clean frames
        print("\nStep 2: Cleaning frames and detecting fashion items...")
        subprocess.run(['python3', 'clean_frames_2.py'], check=True)
        
        # Step 3: Select best frames
        print("\nStep 3: Selecting best fashion frames...")
        subprocess.run(['python3', 'select_best_frames_2.py'], check=True)
        
        print("\nAll processing complete!")
        print(f"1. Extracted frames: {len(os.listdir(FRAMES_DIR))}")
        print(f"2. Clean fashion frames: {len(os.listdir(CLEAN_FRAMES_DIR))}")
        print(f"3. Best fashion frames: {len(os.listdir(BEST_FRAMES_DIR))}")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main() 