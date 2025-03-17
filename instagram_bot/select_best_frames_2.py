import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Maximum number of frames to select per group
MAX_FRAMES_PER_GROUP = 1

def compute_frame_features(image_path):
    """Compute features for a frame using color histogram."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize for consistency
        img = cv2.resize(img, (224, 224))
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def compute_frame_quality(image_path):
    """Compute quality score for a frame."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        
        # Compute sharpness using Laplacian
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Compute brightness
        brightness = np.mean(gray)
        
        # Compute contrast
        contrast = np.std(gray)
        
        # Combine metrics into a single quality score
        quality_score = (sharpness * 0.5 + brightness * 0.25 + contrast * 0.25) / 255.0
        
        return quality_score
    except Exception as e:
        print(f"Error computing quality for {image_path}: {str(e)}")
        return 0

def group_similar_frames(frames_dir):
    """Group similar frames and select the best ones from each group."""
    print("Loading and analyzing fashion frames...")
    
    # Get all image files
    image_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No frames found to process!")
        return []
    
    # Compute features for all frames
    features = []
    valid_files = []
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(frames_dir, img_file)
        feat = compute_frame_features(img_path)
        
        if feat is not None:
            features.append(feat)
            valid_files.append(img_path)
    
    if not valid_files:
        return []
    
    # Convert to numpy array
    features = np.array(features)
    
    # Cluster similar frames
    clustering = DBSCAN(eps=0.3, min_samples=1).fit(features)
    labels = clustering.labels_
    
    # Group frames by cluster
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(valid_files[i])
    
    # Select best frames from each group
    selected_frames = []
    
    for group in groups.values():
        # Sort frames by quality
        frames_with_quality = [(f, compute_frame_quality(f)) for f in group]
        frames_with_quality.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N frames
        for frame, quality in frames_with_quality[:MAX_FRAMES_PER_GROUP]:
            selected_frames.append(frame)
            print(f"Selected frame {frame} with quality {quality:.2f}")
    
    return selected_frames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Select best frames from a directory')
    parser.add_argument('--input', required=True, help='Input directory containing frames')
    
    args = parser.parse_args()
    
    selected = group_similar_frames(args.input)
    print(f"\nSelected {len(selected)} best frames") 