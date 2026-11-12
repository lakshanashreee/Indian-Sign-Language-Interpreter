import os
import sys
import time
import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=== Starting Indian Sign Language Model Training ===", flush=True)

win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

dataset_dir = "dataset_ISL"
classes = sorted(os.listdir(dataset_dir))
print(f"Found {len(classes)} classes: {classes}", flush=True)

X = []
y = []

# To ensure quick training and high quality, load up to 250 diverse images per class
# 250 * 26 = 6,500 samples
MAX_SAMPLES_PER_CLASS = 250
t0 = time.time()

for idx, c in enumerate(classes):
    class_dir = os.path.join(dataset_dir, c)
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Sample evenly
    step = max(1, len(files) // MAX_SAMPLES_PER_CLASS)
    sampled = files[::step][:MAX_SAMPLES_PER_CLASS]
    
    for f in sampled:
        img_path = os.path.join(class_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, win_size)
        feat = hog.compute(resized)
        X.append(feat.flatten())
        y.append(idx)
        
    print(f"Loaded class '{c}' ({len(sampled)} samples)", flush=True)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"\nFeature extraction complete in {time.time()-t0:.2f}s!", flush=True)
print(f"Total dataset shape: X={X.shape}, y={y.shape}", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}", flush=True)

print("Training neural network classifier (MLP)...", flush=True)
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=60,
    batch_size=128,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42,
    verbose=True
)

t_train = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - t_train
print(f"Training completed in {train_time:.2f}s", flush=True)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n==========================================", flush=True)
print(f" ISL Model Evaluation Accuracy: {acc*100:.2f}%", flush=True)
print(f"==========================================\n", flush=True)

# Also test on second dataset (ISL_Dataset) to verify cross-dataset generalization!
if os.path.exists("ISL_Dataset"):
    X2, y2 = [], []
    for c_upper in sorted(os.listdir("ISL_Dataset")):
        c_lower = c_upper.lower()
        if c_lower in classes:
            target_idx = classes.index(c_lower)
            folder = os.path.join("ISL_Dataset", c_upper)
            for f in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                resized = cv2.resize(img, win_size)
                feat = hog.compute(resized)
                X2.append(feat.flatten())
                y2.append(target_idx)
    if X2:
        acc2 = accuracy_score(y2, clf.predict(np.array(X2, dtype=np.float32)))
        print(f"Cross-dataset generalization accuracy on ISL_Dataset: {acc2*100:.2f}%", flush=True)

model_payload = {
    'classifier': clf,
    'classes': [c.upper() for c in classes],
    'win_size': win_size,
    'block_size': block_size,
    'block_stride': block_stride,
    'cell_size': cell_size,
    'nbins': nbins,
    'accuracy': float(acc),
    'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
}

output_path = "isl_model.joblib"
joblib.dump(model_payload, output_path)
print(f"Model successfully saved to: {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)", flush=True)

