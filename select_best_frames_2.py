import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from pathlib import Path

# Directories
input_dir = 'clean_frames_fashion'
output_dir = 'best_fashion_frames'
os.makedirs(output_dir, exist_ok=True)

# Parameters
SIMILARITY_THRESHOLD = 0.80  # More groups for fashion items
MAX_FRAMES_PER_GROUP = 2    # Keep top 2 clearest images from each group

def calculate_fashion_features(image):
    """Calculate features with focus on fashion items"""
    # Resize for consistent comparison
    resized = cv2.resize(image, (224, 224))
    
    # Calculate features from both color and grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Multiple feature types for better fashion comparison
    gray_hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hue_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    sat_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    
    # Normalize and combine features
    gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
    hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()
    sat_hist = cv2.normalize(sat_hist, sat_hist).flatten()
    
    # Combine features with emphasis on color
    features = np.concatenate([gray_hist * 0.3, hue_hist * 0.4, sat_hist * 0.3])
    return features

def calculate_frame_quality(image):
    """Calculate frame quality score"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpness score
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrast score
    contrast = np.std(gray)
    
    # Combined quality score with emphasis on sharpness
    quality_score = (laplacian * 0.7) + (contrast * 0.3)
    
    return quality_score, laplacian, contrast

def group_similar_frames():
    """Group similar fashion frames and select best from each group"""
    print("Loading and analyzing fashion frames...")
    
    frame_data = []
    frame_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(input_dir, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue
            
        features = calculate_fashion_features(image)
        quality_score, sharpness, contrast = calculate_frame_quality(image)
        
        frame_data.append({
            'file': frame_file,
            'features': features,
            'quality_score': quality_score,
            'sharpness': sharpness,
            'contrast': contrast,
            'path': frame_path
        })
    
    if not frame_data:
        print("No frames found to process!")
        return
    
    # Create feature matrix
    feature_matrix = np.array([fd['features'] for fd in frame_data])
    
    # Calculate distance matrix
    distances = np.zeros((len(feature_matrix), len(feature_matrix)))
    for i in range(len(feature_matrix)):
        for j in range(i + 1, len(feature_matrix)):
            distance = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
            distances[i][j] = distance
            distances[j][i] = distance
    
    # Cluster similar frames
    eps = (1 - SIMILARITY_THRESHOLD) * np.max(distances)
    clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(distances)
    
    # Group frames by cluster
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(frame_data[idx])
    
    print(f"\nFound {len(clusters)} groups of similar fashion frames")
    
    # Select best frames from each group
    selected_frames = []
    for group_idx, frames in clusters.items():
        frames.sort(key=lambda x: x['quality_score'], reverse=True)
        top_frames = frames[:MAX_FRAMES_PER_GROUP]
        selected_frames.extend(top_frames)
        
        print(f"\nGroup {group_idx + 1} ({len(frames)} frames):")
        for frame in frames:
            status = "SELECTED" if frame in top_frames else "skipped"
            print(f"  {frame['file']}: quality={frame['quality_score']:.2f}, "
                  f"sharpness={frame['sharpness']:.2f}, "
                  f"contrast={frame['contrast']:.2f} - {status}")
    
    # Save selected frames
    print("\nSaving best fashion frames...")
    for idx, frame in enumerate(selected_frames):
        new_name = f'best_fashion_{idx:03d}.jpg'
        shutil.copy(frame['path'], os.path.join(output_dir, new_name))
        print(f"Saved {frame['file']} as {new_name} "
              f"(quality: {frame['quality_score']:.2f})")
    
    print(f"\nProcessing complete:")
    print(f"Total input frames: {len(frame_files)}")
    print(f"Groups found: {len(clusters)}")
    print(f"Best frames saved: {len(selected_frames)}")
    print(f"Best frames saved in '{output_dir}'")

if __name__ == "__main__":
    group_similar_frames() 