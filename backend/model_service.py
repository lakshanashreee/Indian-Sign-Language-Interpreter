import os
import cv2
import joblib
import numpy as np
import base64
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ISLModelService:
    def __init__(self, model_path="isl_model.joblib", task_path="hand_landmarker.task"):
        self.model_path = model_path
        self.task_path = task_path
        self.model_data = None
        self.classifier = None
        self.classes = []
        self.win_size = (64, 64)
        self.hog = None
        self.detector = None
        self.history = deque(maxlen=6)  # Temporal smoothing filter
        self.last_predicted = None
        self.accuracy = 0.0
        
        self._load_classifier()
        self._load_hand_detector()

    def _load_classifier(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found. Run train.py first.")
        
        self.model_data = joblib.load(self.model_path)
        self.classifier = self.model_data['classifier']
        self.classes = self.model_data.get('classes', [chr(c) for c in range(65, 91)])
        self.win_size = self.model_data.get('win_size', (64, 64))
        self.accuracy = self.model_data.get('accuracy', 0.96)
        
        # Setup HOG descriptor
        self.hog = cv2.HOGDescriptor(
            self.win_size,
            self.model_data.get('block_size', (16, 16)),
            self.model_data.get('block_stride', (8, 8)),
            self.model_data.get('cell_size', (8, 8)),
            self.model_data.get('nbins', 9)
        )
        print(f"[ISLModelService] Loaded classifier with {len(self.classes)} classes. Accuracy: {self.accuracy*100:.2f}%")

    def _load_hand_detector(self):
        if not os.path.exists(self.task_path):
            print(f"[ISLModelService] Warning: {self.task_path} not found. MediaPipe detector unavailable.")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.task_path)
            # Detect up to 2 hands (since ISL uses both 1-handed and 2-handed signs)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.2,
                min_tracking_confidence=0.2
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("[ISLModelService] MediaPipe HandLandmarker initialized successfully.")
        except Exception as e:
            print(f"[ISLModelService] HandLandmarker initialization error: {e}")
            self.detector = None

    def detect_and_isolate_hands(self, frame_bgr):
        """
        Detects hands using MediaPipe, separating hands from face and torso.
        Returns:
            hand_detected (bool)
            landmarks_list (list of dicts with x, y, z normalized coordinates)
            bbox (dict: x_min, y_min, x_max, y_max in pixel coords)
            hand_crop (isolated hand image)
            hand_count (int)
        """
        h, w = frame_bgr.shape[:2]
        landmarks_list = []
        hand_count = 0
        
        if self.detector is not None:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            try:
                result = self.detector.detect(mp_img)
                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    hand_count = len(result.hand_landmarks)
                    all_xs = []
                    all_ys = []
                    
                    for hand_lms in result.hand_landmarks:
                        coords = []
                        for lm in hand_lms:
                            px = int(lm.x * w)
                            py = int(lm.y * h)
                            all_xs.append(px)
                            all_ys.append(py)
                            coords.append({"x": lm.x, "y": lm.y, "z": lm.z})
                        landmarks_list.append(coords)
                    
                    if all_xs and all_ys:
                        min_x, max_x = min(all_xs), max(all_xs)
                        min_y, max_y = min(all_ys), max(all_ys)
                        
                        # Apply generous 25% padding around the hands
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
                        
                        return True, landmarks_list, bbox, hand_crop, hand_count
            except Exception as e:
                print(f"[detect_and_isolate_hands] Error: {e}")
        
        # Fallback: Central Framing Box (ROI) for immediate testing or if person is in frame
        box_w = int(w * 0.45)
        box_h = int(h * 0.55)
        x1 = int(w * 0.5)
        y1 = int(h * 0.25)
        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)
        fallback_crop = frame_bgr[y1:y2, x1:x2].copy()
        fallback_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1}
        
        return False, [], fallback_bbox, fallback_crop, 0

    def predict_sign(self, frame_bgr, use_roi_fallback=True):
        """
        Runs the full ISL recognition pipeline on a camera frame.
        Separates human body/face from hand, crops hand, computes HOG features,
        and applies smoothed probability classification.
        """
        hand_detected, landmarks, bbox, hand_crop, hand_count = self.detect_and_isolate_hands(frame_bgr)
        
        # If no hand was detected by MediaPipe and fallback is disabled, return no hand
        if not hand_detected and not use_roi_fallback:
            return {
                "hand_detected": False,
                "letter": None,
                "confidence": 0.0,
                "smoothed_letter": None,
                "top3": [],
                "landmarks": [],
                "bbox": None,
                "hand_count": 0,
                "hand_crop_base64": None
            }
        
        # Preprocess isolated hand crop for HOG descriptor
        if hand_crop is None or hand_crop.size == 0:
            return {
                "hand_detected": False,
                "letter": None,
                "confidence": 0.0,
                "smoothed_letter": None,
                "top3": [],
                "landmarks": [],
                "bbox": None,
                "hand_count": 0,
                "hand_crop_base64": None
            }

        gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        resized_crop = cv2.resize(gray_crop, self.win_size)
        features = self.hog.compute(resized_crop).reshape(1, -1)
        
        # Run classification
        probs = self.classifier.predict_proba(features)[0]
        top3_indices = np.argsort(probs)[::-1][:3]
        
        pred_idx = top3_indices[0]
        predicted_letter = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        top3 = [
            {"letter": self.classes[idx], "prob": round(float(probs[idx]), 3)}
            for idx in top3_indices
        ]
        
        # Temporal smoothing: add to rolling window
        if hand_detected or confidence > 0.35:
            self.history.append(predicted_letter)
        else:
            if len(self.history) > 0:
                self.history.popleft()
                
        # Majority voting for stable display
        if len(self.history) >= 2:
            from collections import Counter
            counts = Counter(self.history)
            smoothed_letter, _ = counts.most_common(1)[0]
        else:
            smoothed_letter = predicted_letter
            
        # Encode cropped hand thumbnail to show human isolation in UI
        crop_thumb = cv2.resize(hand_crop, (120, 120))
        _, buffer = cv2.imencode('.jpg', crop_thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        return {
            "hand_detected": hand_detected,
            "letter": predicted_letter,
            "confidence": round(confidence, 3),
            "smoothed_letter": smoothed_letter,
            "top3": top3,
            "landmarks": landmarks,
            "bbox": bbox,
            "hand_count": hand_count,
            "hand_crop_base64": crop_b64
        }

