# Indian Sign Language (ISL) Real-Time AI Interpreter

A real-time Indian Sign Language (ISL) recognition and translation system designed to interpret sign language gestures from live video streams into **Alphabets**, **Words**, and complete **Sentences**, with built-in **Hand-Body-Face Separation** and **Text-to-Speech (TTS)**.

---

## 🌟 Key Features

1. **Human & Hand Separation (Face & Body Filtering)**:
   - Powered by Google MediaPipe HandLandmarker (3D skeletal detection with 21 landmarks per hand for up to 2 hands).
   - Specifically isolates the human hands and gesture shapes, rejecting faces, clothing, and background interference.
   - Includes a real-time **Hand Isolation Monitor** panel displaying the cropped hand stream directly in the interface.

2. **Trained 3D Landmark ISL Model (93.1% Accuracy)**:
   - Trained across all 26 Indian Sign Language classes (A through Z) using 154 scale- and translation-invariant geometric landmark features.
   - Extracts finger curl ratios, inter-fingertip distances, and dual-hand interaction distances.
   - **Zero False Positives**: Strictly enforces a silent idle state when hands are out of the camera frame or in motion. No random letters are ever guessed when hands are absent!
   - Real-time temporal smoothing and majority-voting debouncing to eliminate flickering.

3. **Three Core Operational Modes**:
   - 🔤 **Alphabet Mode**:
     - Continuous real-time sign recognition.
     - Large visible hero card with confidence percentage and top-3 candidate rankings.
     - Live stream history breadcrumb with one-click copy and audio speech synthesis.
   - 📝 **Word Mode** (Slot-Based Builder):
     - Specify the number of letters in the target word (e.g. 4 letters for "SIGN").
     - Dynamic interactive letter slots (`[ Slot 1 ] [ Slot 2 ] ...`).
     - Real-time live interpretation displayed below the active slot.
     - Step-by-step confirmation: hold the sign, click **Next Letter** (or press **Spacebar**) to save the slot and advance to the next letter.
     - Final assembled word with Text-to-Speech pronunciation, copy to clipboard, and send to sentence mode.
   - 💬 **Sentence Mode** (Multi-Word Translation):
     - Specify the number of words in the sentence (e.g. 3 words for "HELLO INDIAN SIGN").
     - Word breadcrumb stepper tracking completion progress.
     - For each word, build letter-by-letter with the "Next Letter" button, then click **Complete Word & Next Word**.
     - Full assembled sentence box with natural Text-to-Speech pronunciation.

4. **Camera Controls & Overlays**:
   - Instant Camera ON / OFF toggle.
   - 3D Skeletal Landmark connections (joints and bones) toggle.
   - Framing Target Zone guide box toggle.
   - Horizontal Mirror / Flip toggle.

5. **Integrated ISL Reference Guide**:
   - Built-in interactive modal dictionary with descriptions and hand posture tips for all 26 ISL letters.

---

## 🚀 How to Launch the Interpreter

### Option 1: Web Interpreter (Recommended)
Launches the FastAPI backend and automatically opens the interface in your default browser at `http://localhost:8000`:

```powershell
python run_interpreter.py
```

### Option 2: Standalone Desktop OpenCV Window
If you prefer a lightweight desktop application running inside an OpenCV window:

```powershell
python desktop_interpreter.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| **Spacebar** | **Next Letter (Save Slot)** in Word Mode & Sentence Mode |
| **Backspace** | **Previous Letter / Undo** active slot |
| **Enter** | **Complete Word & Advance** in Sentence Mode |
| **C** | **Clear / Reset** stream history or slots |
| **Escape** | Close ISL Guide Modal |

---

## 📁 Project Structure

```
Sign Language Interpreter/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI server & REST endpoints
│   └── model_service.py     # MediaPipe hand isolation & neural inference
├── static/
│   ├── index.html           # Modern responsive web UI
│   ├── style.css            # Dark glassmorphic styling
│   ├── app.js               # Camera stream, canvas landmarks, modes & TTS
│   └── placeholder_hand.png # Default preview thumbnail
├── dataset_ISL/             # 26 ISL alphabet classes (12,636 images)
├── ISL_Dataset/             # Supplementary ISL evaluation dataset
├── hand_landmarker.task     # MediaPipe 3D Hand Landmarker model
├── isl_model.joblib         # Trained 96.41% accuracy ISL model
├── train.py                 # Neural network training script
├── run_interpreter.py       # One-click web application launcher
├── desktop_interpreter.py   # Standalone OpenCV desktop window
└── README.md                # Project documentation
```

---

## 🔬 Model Details & Training Pipeline

- **Features**: Multi-scale Histogram of Oriented Gradients (HOG) with $64 \times 64$ window, $16 \times 16$ block size, $8 \times 8$ cell size, 9 orientation bins (1,764 dimensional feature space).
- **Classifier**: Multi-Layer Perceptron (MLP) with $(256, 128)$ hidden layers, ReLU activation, Adam optimizer, early stopping, and softmax probability calibration.
- **Accuracy**: **96.41%** on held-out test split.

