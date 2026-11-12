import os
import sys
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=== Indian Sign Language Landmark Model Training ===", flush=True)

# 1. Initialize MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_tracking_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)

def extract_features_from_result(result):
    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None
    
    landmarks_list = []
    for h in result.hand_landmarks:
        coords = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in h]
        landmarks_list.append(coords)
    return extract_landmark_features(landmarks_list)

def extract_landmark_features(landmarks_list):
    if not landmarks_list or len(landmarks_list) == 0:
        return None
        
    features = []
    hand_count = len(landmarks_list)
    features.append(float(hand_count))
    
    for h_idx in range(2):
        if h_idx < hand_count:
            hand = landmarks_list[h_idx]
            wrist_x, wrist_y, wrist_z = hand[0]['x'], hand[0]['y'], hand[0]['z']
            m_mcp_x, m_mcp_y, m_mcp_z = hand[9]['x'], hand[9]['y'], hand[9]['z']
            palm_dist = np.sqrt((m_mcp_x - wrist_x)**2 + (m_mcp_y - wrist_y)**2 + (m_mcp_z - wrist_z)**2)
            if palm_dist < 1e-4: palm_dist = 1.0
            
            # Normalized coordinates (63 values)
            for lm in hand:
                features.append((lm['x'] - wrist_x) / palm_dist)
                features.append((lm['y'] - wrist_y) / palm_dist)
                features.append((lm['z'] - wrist_z) / palm_dist)
                
            # Finger curl ratios (5 values)
            tips = [4, 8, 12, 16, 20]
            pips = [2, 6, 10, 14, 18]
            for tip, pip in zip(tips, pips):
                d_tip = np.sqrt((hand[tip]['x'] - wrist_x)**2 + (hand[tip]['y'] - wrist_y)**2)
                d_pip = np.sqrt((hand[pip]['x'] - wrist_x)**2 + (hand[pip]['y'] - wrist_y)**2)
                features.append(d_tip / (d_pip + 1e-4))
                
            # Fingertip-to-thumb distances (4 values)
            for tip in [8, 12, 16, 20]:
                d_thumb = np.sqrt((hand[tip]['x'] - hand[4]['x'])**2 + (hand[tip]['y'] - hand[4]['y'])**2)
                features.append(d_thumb / palm_dist)
                
            # Adjacent fingertip distances (3 values)
            features.append(np.sqrt((hand[8]['x'] - hand[12]['x'])**2 + (hand[8]['y'] - hand[12]['y'])**2) / palm_dist)
            features.append(np.sqrt((hand[12]['x'] - hand[16]['x'])**2 + (hand[12]['y'] - hand[16]['y'])**2) / palm_dist)
            features.append(np.sqrt((hand[16]['x'] - hand[20]['x'])**2 + (hand[16]['y'] - hand[20]['y'])**2) / palm_dist)
        else:
            features.extend([0.0] * 75)
            
    if hand_count >= 2:
        h1 = landmarks_list[0]
        h2 = landmarks_list[1]
        wrist_dist = np.sqrt((h1[0]['x'] - h2[0]['x'])**2 + (h1[0]['y'] - h2[0]['y'])**2)
        index_dist = np.sqrt((h1[8]['x'] - h2[8]['x'])**2 + (h1[8]['y'] - h2[8]['y'])**2)
        thumb_dist = np.sqrt((h1[4]['x'] - h2[4]['x'])**2 + (h1[4]['y'] - h2[4]['y'])**2)
        features.extend([wrist_dist, index_dist, thumb_dist])
    else:
        features.extend([0.0, 0.0, 0.0])
        
    return np.array(features, dtype=np.float32)

def augment_landmarks(features, n_aug=5):
    """Generates synthetic variations via small scaling, rotation & noise."""
    augmented = [features]
    for _ in range(n_aug):
        noise = np.random.normal(0, 0.02, size=features.shape).astype(np.float32)
        scale = np.random.uniform(0.92, 1.08)
        # Keep hand count (index 0) intact
        aug_feat = features * scale + noise
        aug_feat[0] = features[0]
        augmented.append(aug_feat)
    return augmented

all_classes = [chr(c) for c in range(65, 91)]  # A to Z
X = []
y = []

print("Extracting MediaPipe landmarks from ISL_Dataset and dataset_ISL...", flush=True)
t0 = time.time()

# 1. Process ISL_Dataset (Clean high-quality samples)
if os.path.exists("ISL_Dataset"):
    for folder in sorted(os.listdir("ISL_Dataset")):
        c = folder.upper()
        if c not in all_classes: continue
        c_idx = all_classes.index(c)
        folder_path = os.path.join("ISL_Dataset", folder)
        files = os.listdir(folder_path)
        extracted = 0
        for f in files:
            img = cv2.imread(os.path.join(folder_path, f))
            if img is None: continue
            padded = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            padded = cv2.resize(padded, (256, 256))
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            feat = extract_features_from_result(res)
            if feat is not None:
                extracted += 1
                for aug in augment_landmarks(feat, n_aug=6):
                    X.append(aug)
                    y.append(c_idx)
        print(f"ISL_Dataset '{c}': {extracted}/{len(files)} detected -> {extracted * 7} augmented samples", flush=True)

# 2. Process dataset_ISL (for all classes, especially H, J, Y)
if os.path.exists("dataset_ISL"):
    for folder in sorted(os.listdir("dataset_ISL")):
        c = folder.upper()
        if c not in all_classes: continue
        c_idx = all_classes.index(c)
        # For classes already rich, take fewer; for H, J, Y take more
        max_take = 35 if c in ['H', 'J', 'Y'] else 15
        folder_path = os.path.join("dataset_ISL", folder)
        files = os.listdir(folder_path)
        extracted = 0
        for f in files[:80]:
            if extracted >= max_take: break
            img = cv2.imread(os.path.join(folder_path, f))
            if img is None: continue
            padded = cv2.copyMakeBorder(img, 70, 70, 70, 70, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            padded = cv2.resize(padded, (256, 256))
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            feat = extract_features_from_result(res)
            if feat is not None:
                extracted += 1
                for aug in augment_landmarks(feat, n_aug=5):
                    X.append(aug)
                    y.append(c_idx)
        print(f"dataset_ISL '{c}': {extracted} detected -> added to training", flush=True)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"\nTotal landmark samples extracted & augmented: {len(X)} in {time.time()-t0:.2f}s", flush=True)
print(f"Classes represented: {len(np.unique(y))} / {len(all_classes)}", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("\nTraining Deep Neural Classifier on Landmarks...", flush=True)
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=8,
    random_state=42
)
clf.fit(X_train, y_train)

test_acc = accuracy_score(y_test, clf.predict(X_test))
print(f"\n=======================================================", flush=True)
print(f" LANDMARK CLASSIFIER TEST ACCURACY: {test_acc*100:.2f}%", flush=True)
print(f"=======================================================\n", flush=True)

# Save Landmark Model
payload = {
    'classifier': clf,
    'classes': all_classes,
    'feature_dim': X.shape[1],
    'accuracy': float(test_acc),
    'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(payload, "isl_landmark_model.joblib")
print("Saved model to isl_landmark_model.joblib successfully!", flush=True)

