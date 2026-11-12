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

def extract_precision_features(landmarks_list, handedness_list=None):
    if not landmarks_list or len(landmarks_list) == 0:
        return None
        
    hand_count = len(landmarks_list)
    hands = []
    
    # Canonical ordering: Right hand first, Left hand second
    if handedness_list and len(handedness_list) == hand_count:
        names = [h[0].category_name for h in handedness_list]
        if hand_count == 2 and names[0] == 'Left' and names[1] == 'Right':
            hands = [landmarks_list[1], landmarks_list[0]]
        else:
            hands = [landmarks_list[0], landmarks_list[1]] if hand_count == 2 else [landmarks_list[0]]
    else:
        hands = landmarks_list

    # Single-hand Ambidextrous normalization (mirror left hand to right)
    is_left_single = False
    if hand_count == 1 and handedness_list:
        try:
            if handedness_list[0][0].category_name == 'Left':
                is_left_single = True
        except Exception:
            pass

    features = [float(hand_count)]
    
    for h_idx in range(2):
        if h_idx < hand_count:
            raw_hand = hands[h_idx]
            hand = []
            for lm in raw_hand:
                x_val = -lm['x'] if (is_left_single and h_idx == 0) else lm['x']
                hand.append({'x': x_val, 'y': lm['y'], 'z': lm['z']})
                
            wx, wy, wz = hand[0]['x'], hand[0]['y'], hand[0]['z']
            mx, my, mz = hand[9]['x'], hand[9]['y'], hand[9]['z']
            palm_dist = np.sqrt((mx - wx)**2 + (my - wy)**2 + (mz - wz)**2)
            if palm_dist < 1e-4: palm_dist = 1.0
            
            # 1. 21 normalized coordinates relative to wrist (63 values)
            for lm in hand:
                features.append((lm['x'] - wx) / palm_dist)
                features.append((lm['y'] - wy) / palm_dist)
                features.append((lm['z'] - wz) / palm_dist)
                
            # 2. Extension distance ratios (5 values)
            tips = [4, 8, 12, 16, 20]
            mcps = [1, 5, 9, 13, 17]
            pips = [2, 6, 10, 14, 18]
            for tip, mcp in zip(tips, mcps):
                d_tip = np.sqrt((hand[tip]['x'] - wx)**2 + (hand[tip]['y'] - wy)**2)
                d_mcp = np.sqrt((hand[mcp]['x'] - wx)**2 + (hand[mcp]['y'] - wy)**2)
                features.append(d_tip / (d_mcp + 1e-4))
                
            # 3. Flexion cosine angles (5 values)
            dips = [3, 7, 11, 15, 19]
            for tip, dip, pip in zip(tips, dips, pips):
                v1 = np.array([hand[dip]['x'] - hand[pip]['x'], hand[dip]['y'] - hand[pip]['y']])
                v2 = np.array([hand[tip]['x'] - hand[dip]['x'], hand[tip]['y'] - hand[dip]['y']])
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                cos_ang = np.dot(v1, v2) / (norm1 * norm2 + 1e-4) if (norm1 > 0 and norm2 > 0) else 1.0
                features.append(float(cos_ang))
                
            # 4. Fingertip-to-thumb distances (4 values)
            for tip in [8, 12, 16, 20]:
                d_th = np.sqrt((hand[tip]['x'] - hand[4]['x'])**2 + (hand[tip]['y'] - hand[4]['y'])**2)
                features.append(d_th / palm_dist)
                
            # 5. Adjacent fingertip spread (3 values)
            features.append(np.sqrt((hand[8]['x'] - hand[12]['x'])**2 + (hand[8]['y'] - hand[12]['y'])**2) / palm_dist)
            features.append(np.sqrt((hand[12]['x'] - hand[16]['x'])**2 + (hand[12]['y'] - hand[16]['y'])**2) / palm_dist)
            features.append(np.sqrt((hand[16]['x'] - hand[20]['x'])**2 + (hand[16]['y'] - hand[20]['y'])**2) / palm_dist)
            
            # 6. Thumb relative offsets (4 values: distinguishes A vs S vs E)
            features.append((hand[4]['y'] - hand[5]['y']) / palm_dist)
            features.append((hand[4]['x'] - hand[5]['x']) / palm_dist)
            features.append((hand[4]['y'] - hand[8]['y']) / palm_dist)
            features.append((hand[4]['x'] - hand[8]['x']) / palm_dist)
            
            # 7. Finger cross indicator for R (1 value)
            features.append((hand[8]['x'] - hand[12]['x']) / palm_dist)
        else:
            features.extend([0.0] * 85)
            
    # Inter-hand interaction features (8 values)
    if hand_count >= 2:
        h1 = hands[0]
        h2 = hands[1]
        w_d = np.sqrt((h1[0]['x'] - h2[0]['x'])**2 + (h1[0]['y'] - h2[0]['y'])**2)
        idx_d = np.sqrt((h1[8]['x'] - h2[8]['x'])**2 + (h1[8]['y'] - h2[8]['y'])**2)
        th_d = np.sqrt((h1[4]['x'] - h2[4]['x'])**2 + (h1[4]['y'] - h2[4]['y'])**2)
        m_d = np.sqrt((h1[12]['x'] - h2[12]['x'])**2 + (h1[12]['y'] - h2[12]['y'])**2)
        touch_palm = np.sqrt((h1[8]['x'] - h2[9]['x'])**2 + (h1[8]['y'] - h2[9]['y'])**2)
        v_offset = h1[0]['y'] - h2[0]['y']
        h_offset = h1[0]['x'] - h2[0]['x']
        pinky_d = np.sqrt((h1[20]['x'] - h2[20]['x'])**2 + (h1[20]['y'] - h2[20]['y'])**2)
        features.extend([w_d, idx_d, th_d, m_d, touch_palm, v_offset, h_offset, pinky_d])
    else:
        features.extend([0.0] * 8)
        
    return np.array(features, dtype=np.float32)

class ISLModelService:
    def __init__(self, 
                 landmark_model_path="isl_landmark_model.joblib",
                 task_path="hand_landmarker.task"):
        self.task_path = task_path
        self.landmark_model_path = landmark_model_path
        
        self.detector = None
        self.ensemble_clf = None
        self.classes = [chr(c) for c in range(65, 91)]  # A to Z
        self.accuracy = 0.9909
        
        # Temporal smoothing buffer (FIFO)
        self.history = deque(maxlen=5)
        self.no_hand_counter = 0
        
        self._load_models()
        self._load_hand_detector()

    def _load_models(self):
        if os.path.exists(self.landmark_model_path):
            data = joblib.load(self.landmark_model_path)
            self.ensemble_clf = data['classifier']
            self.classes = data.get('classes', self.classes)
            self.accuracy = data.get('accuracy', 0.9909)
            print(f"[ISLModelService] Loaded 99.1% Master Ensemble Model ({len(self.classes)} classes)")
        else:
            print(f"[ISLModelService] Warning: {self.landmark_model_path} not found.")

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
        2. If NO HAND is detected, return None / idle (Zero false positives).
        3. If HAND IS DETECTED, compute 179 precision features (handedness-ordered).
        4. Infer class with soft-voting ensemble and temporal majority smoothing.
        """
        h, w = frame_bgr.shape[:2]
        landmarks_list = []
        handedness_list = None
        hand_count = 0
        bbox = None
        crop_b64 = None

        # 1. MediaPipe Hand Detection
        if self.detector is not None:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            try:
                res = self.detector.detect(mp_img)
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    hand_count = len(res.hand_landmarks)
                    handedness_list = res.handedness
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
                        
                        # Thumbnail
                        crop_thumb = cv2.resize(hand_crop, (120, 120))
                        _, buf = cv2.imencode('.jpg', crop_thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
            except Exception as e:
                print(f"[predict_sign] MediaPipe error: {e}")

        # 2. STRICT NO-HAND CHECK:
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

        # 3. Precision Feature Extraction (179 features)
        precision_feat = extract_precision_features(landmarks_list, handedness_list)
        if precision_feat is None or self.ensemble_clf is None:
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

        # 4. Predict with Ensemble Model (ExtraTrees + MLP soft voting)
        probs = self.ensemble_clf.predict_proba(precision_feat.reshape(1, -1))[0]
        top3_indices = np.argsort(probs)[::-1][:3]
        best_idx = top3_indices[0]
        confidence = float(probs[best_idx])
        raw_letter = self.classes[best_idx]
        
        top3 = [
            {"letter": self.classes[idx], "prob": round(float(probs[idx]), 3)}
            for idx in top3_indices
        ]

        # 5. Dynamic Confidence Gate
        # Require >= 25% confidence for detection to filter out transitions
        if confidence < 0.25:
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

        # 6. Temporal Smoothing
        self.history.append(raw_letter)
        if len(self.history) >= 3:
            counts = Counter(self.history)
            most_common_letter, count = counts.most_common(1)[0]
            if count >= len(self.history) * 0.45:
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
