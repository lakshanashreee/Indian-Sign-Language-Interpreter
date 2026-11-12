import os
import cv2
import joblib
import numpy as np
import base64
import time
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_landmark_features(landmarks_list):
    """
    Extracts 154 scale-, position-, and lighting-invariant geometric features
    from MediaPipe hand landmarks.
    """
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
            
            # 1. 21 normalized 3D coordinates relative to wrist (63 values)
            for lm in hand:
                features.append((lm['x'] - wrist_x) / palm_dist)
                features.append((lm['y'] - wrist_y) / palm_dist)
                features.append((lm['z'] - wrist_z) / palm_dist)
                
            # 2. Finger curl ratios for all 5 fingers (5 values)
            tips = [4, 8, 12, 16, 20]
            pips = [2, 6, 10, 14, 18]
            for tip, pip in zip(tips, pips):
                d_tip = np.sqrt((hand[tip]['x'] - wrist_x)**2 + (hand[tip]['y'] - wrist_y)**2)
                d_pip = np.sqrt((hand[pip]['x'] - wrist_x)**2 + (hand[pip]['y'] - wrist_y)**2)
                features.append(d_tip / (d_pip + 1e-4))
                
            # 3. Fingertip-to-thumb distances (4 values)
            for tip in [8, 12, 16, 20]:
                d_thumb = np.sqrt((hand[tip]['x'] - hand[4]['x'])**2 + (hand[tip]['y'] - hand[4]['y'])**2)
                features.append(d_thumb / palm_dist)
                
            # 4. Adjacent fingertip distances (3 values)
            features.append(np.sqrt((hand[8]['x'] - hand[12]['x'])**2 + (hand[8]['y'] - hand[12]['y'])**2) / palm_dist)
            features.append(np.sqrt((hand[12]['x'] - hand[16]['x'])**2 + (hand[12]['y'] - hand[16]['y'])**2) / palm_dist)
            features.append(np.sqrt((hand[16]['x'] - hand[20]['x'])**2 + (hand[16]['y'] - hand[20]['y'])**2) / palm_dist)
        else:
            # Zero-padding for second hand when only 1 hand is present (75 zeros)
            features.extend([0.0] * 75)
            
    # Inter-hand features if 2 hands are detected (critical for 2-handed ISL gestures)
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

class ISLModelService:
    def __init__(self, 
                 landmark_model_path="isl_landmark_model.joblib",
                 visual_model_path="isl_model.joblib",
                 task_path="hand_landmarker.task"):
        self.task_path = task_path
        self.landmark_model_path = landmark_model_path
        self.visual_model_path = visual_model_path
        
        self.detector = None
        self.landmark_clf = None
        self.visual_clf = None
        self.classes = [chr(c) for c in range(65, 91)]  # A to Z
        self.accuracy = 0.93
        self.win_size = (64, 64)
        self.hog = None
        
        # Temporal smoothing buffer (FIFO)
        self.history = deque(maxlen=6)
        self.no_hand_counter = 0
        
        self._load_models()
        self._load_hand_detector()

    def _load_models(self):
        # 1. Load Landmark Classifier
        if os.path.exists(self.landmark_model_path):
            data = joblib.load(self.landmark_model_path)
            self.landmark_clf = data['classifier']
            self.classes = data.get('classes', self.classes)
            self.accuracy = data.get('accuracy', 0.93)
            print(f"[ISLModelService] Loaded Landmark Model ({len(self.classes)} classes, {self.accuracy*100:.1f}% Acc)")
        else:
            print(f"[ISLModelService] Landmark model {self.landmark_model_path} not found.")

        # 2. Load Visual HOG Classifier (as ensemble backup)
        if os.path.exists(self.visual_model_path):
            v_data = joblib.load(self.visual_model_path)
            self.visual_clf = v_data['classifier']
            self.hog = cv2.HOGDescriptor(
                self.win_size,
                v_data.get('block_size', (16, 16)),
                v_data.get('block_stride', (8, 8)),
                v_data.get('cell_size', (8, 8)),
                v_data.get('nbins', 9)
            )
            print(f"[ISLModelService] Loaded Visual Model as secondary ensemble.")

    def _load_hand_detector(self):
        if not os.path.exists(self.task_path):
            print(f"[ISLModelService] Warning: {self.task_path} not found.")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.task_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.25,
                min_tracking_confidence=0.25
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("[ISLModelService] MediaPipe HandLandmarker initialized.")
        except Exception as e:
            print(f"[ISLModelService] MediaPipe init error: {e}")
            self.detector = None

    def predict_sign(self, frame_bgr, use_roi_fallback=False):
        """
        Main recognition pipeline:
        1. MediaPipe isolates hands from body and face.
        2. If NO HAND is detected, return None / idle (NO RANDOM LETTERS!).
        3. If HAND IS DETECTED, compute landmark features + isolated hand crop.
        4. Infer class with confidence and temporal smoothing.
        """
        h, w = frame_bgr.shape[:2]
        landmarks_list = []
        hand_count = 0
        bbox = None
        hand_crop = None
        crop_b64 = None

        # 1. MediaPipe Hand Detection
        if self.detector is not None:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            try:
                res = self.detector.detect(mp_img)
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    hand_count = len(res.hand_landmarks)
                    all_xs = []
                    all_ys = []
                    for hand in res.hand_landmarks:
                        coords = []
                        for lm in hand:
                            px = int(lm.x * w)
                            py = int(lm.y * h)
                            all_xs.append(px)
                            all_ys.append(py)
                            coords.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                        landmarks_list.append(coords)
                    
                    if all_xs and all_ys:
                        min_x, max_x = min(all_xs), max(all_xs)
                        min_y, max_y = min(all_ys), max(all_ys)
                        bw = max_x - min_x
                        bh = max_y - min_y
                        pad_x = int(bw * 0.25) + 20
                        pad_y = int(bh * 0.25) + 20
                        x1 = max(0, min_x - pad_x)
                        y1 = max(0, min_y - pad_y)
                        x2 = min(w, max_x + pad_x)
                        y2 = min(h, max_y + pad_y)
                        
                        hand_crop = frame_bgr[y1:y2, x1:x2].copy()
                        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1}
            except Exception as e:
                print(f"[predict_sign] MediaPipe error: {e}")

        # 2. STRICT NO-HAND CHECK:
        # If no hand was detected by MediaPipe, DO NOT PREDICT ANY LETTER!
        if hand_count == 0 or not landmarks_list:
            self.no_hand_counter += 1
            if self.no_hand_counter > 2:
                self.history.clear()
                
            return {
                "hand_detected": False,
                "letter": None,
                "smoothed_letter": "-",
                "confidence": 0.0,
                "top3": [],
                "landmarks": [],
                "bbox": None,
                "hand_count": 0,
                "hand_crop_base64": None
            }

        self.no_hand_counter = 0

        # Encode cropped hand thumbnail
        if hand_crop is not None and hand_crop.size > 0:
            crop_thumb = cv2.resize(hand_crop, (120, 120))
            _, buf = cv2.imencode('.jpg', crop_thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        # 3. Landmark Feature Extraction & Inference
        landmark_feat = extract_landmark_features(landmarks_list)
        if landmark_feat is None or self.landmark_clf is None:
            return {
                "hand_detected": True,
                "letter": None,
                "smoothed_letter": "-",
                "confidence": 0.0,
                "top3": [],
                "landmarks": landmarks_list,
                "bbox": bbox,
                "hand_count": hand_count,
                "hand_crop_base64": crop_b64
            }

        # Predict with Landmark Model
        probs_lm = self.landmark_clf.predict_proba(landmark_feat.reshape(1, -1))[0]
        
        # If visual classifier is also available, compute ensemble
        final_probs = probs_lm
        if self.visual_clf is not None and hand_crop is not None and hand_crop.size > 0:
            try:
                gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                resized_crop = cv2.resize(gray_crop, self.win_size)
                hog_feat = self.hog.compute(resized_crop).reshape(1, -1)
                probs_vis = self.visual_clf.predict_proba(hog_feat)[0]
                # Weighted fusion: 75% Landmark geometry (lighting/skin invariant) + 25% Visual edge
                final_probs = 0.75 * probs_lm + 0.25 * probs_vis
            except Exception:
                final_probs = probs_lm

        top3_indices = np.argsort(final_probs)[::-1][:3]
        best_idx = top3_indices[0]
        confidence = float(final_probs[best_idx])
        raw_letter = self.classes[best_idx]
        
        top3 = [
            {"letter": self.classes[idx], "prob": round(float(final_probs[idx]), 3)}
            for idx in top3_indices
        ]

        # 4. Confidence Threshold Gate
        # If confidence is lower than 35%, hand is likely moving or in transition
        if confidence < 0.35:
            return {
                "hand_detected": True,
                "letter": None,
                "smoothed_letter": "-",
                "confidence": round(confidence, 3),
                "top3": top3,
                "landmarks": landmarks_list,
                "bbox": bbox,
                "hand_count": hand_count,
                "hand_crop_base64": crop_b64
            }

        # 5. Temporal Smoothing Filter
        self.history.append(raw_letter)
        if len(self.history) >= 3:
            counts = Counter(self.history)
            most_common_letter, count = counts.most_common(1)[0]
            # Must appear in at least 50% of recent frames to be accepted
            if count >= len(self.history) * 0.5:
                smoothed_letter = most_common_letter
            else:
                smoothed_letter = raw_letter
        else:
            smoothed_letter = raw_letter

        return {
            "hand_detected": True,
            "letter": raw_letter,
            "smoothed_letter": smoothed_letter,
            "confidence": round(confidence, 3),
            "top3": top3,
            "landmarks": landmarks_list,
            "bbox": bbox,
            "hand_count": hand_count,
            "hand_crop_base64": crop_b64
        }
