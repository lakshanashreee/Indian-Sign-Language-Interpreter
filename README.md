# Indian Sign Language (ISL) Real-Time AI Interpreter
### 3D Kinematic Hand Isolation, Zero False-Positive Gating & Hierarchical Translation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-orange.svg)](https://developers.google.com/mediapipe)
[![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble%20Voting-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous, real-time Indian Sign Language (ISL) interpretation and translation platform designed to translate live sign language gestures from standard webcams into **Alphabets**, **Words**, and **Sentences**, with **Human Hand-Body-Face Separation**, **Zero False Positives**, and **Synchronous Text-to-Speech (TTS)** vocalization.

> 📚 **Complete Technical Dossier & International Forum Presentation Guide**: See [INTERNATIONAL_FORUM_GUIDE.md](INTERNATIONAL_FORUM_GUIDE.md) for full mathematical formulations, competitive differentiations, architectural dataflows, and verbal speech scripts.

---

## 🌟 Key Architectural Highlights

1. **Human Hand-Body-Face Isolation**:
   - Powered by **Google MediaPipe HandLandmarker LiteRT**, capturing 21 3D anatomical skeletal landmarks per hand (up to 42 landmarks for bilateral signs).
   - Rejects the user's face, neck, clothing, skin-tone variations, and background room clutter.
   - Includes a real-time **Hand Crop Isolation Monitor** in the UI.

2. **Strict Zero False-Positive Gating**:
   - Traditional computer vision classifiers force a prediction even when hands are out of the frame or resting at the user's side.
   - Our system enforces a strict anatomical hand-presence gate: **Hands absent = 100% silent, zero hallucinated letters**.

3. **179-Dimensional Invariant Geometric Descriptors**:
   - Invariant to distance from camera (scale), position in frame (translation), and left-to-right signing (ambidextrous reflection symmetry).
   - Captures palm-normalized coordinates, finger extension ratios, joint flexion cosines, and dual-hand interaction matrices.

4. **Hybrid Soft-Voting Classifier Ensemble**:
   - Combines **ExtraTreesClassifier** (150 unpruned randomized trees for sharp boundary separation) with a **Multi-Layer Perceptron (MLP)** neural network (256, 128 hidden layers for continuous spatial manifolds).
   - Achieves **97.57% test accuracy** across all 26 ISL classes.

5. **Hierarchical Linguistic Discourse Modes**:
   - 🔤 **Alphabet Mode**: Real-time continuous sign recognition with high-contrast hero typography, confidence score, and top-3 candidate rankings.
   - 📝 **Word Mode** (Dynamic Slot-Based Builder): Define word length $\to$ dynamic letter slots $\to$ live interpretation under active slot $\to$ press `Spacebar` to lock and advance $\to$ automatic Text-to-Speech vocalization.
   - 💬 **Sentence Mode** (Multi-Word Translation): Multi-word navigation stepper $\to$ letter-by-letter composition $\to$ `Enter` to commit words $\to$ full grammatical sentence speech synthesis and clipboard copy.

6. **Zero-Cloud, Privacy-First Edge Inference**:
   - Runs 100% locally on standard consumer CPUs at 30+ FPS.
   - Zero GPU requirement, zero cloud API costs, air-gapped security.

---

## 🚀 Quick Start & Launch

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.
- Standard webcam or USB camera.

### 1. Automated Installation
Run the dependency installer batch script (or install via pip):
```powershell
.\install_dependencies.bat
```
Or manually:
```powershell
pip install -r requirements.txt
```

### 2. Launching the Application

#### Option A: Web Application (Recommended)
Double-click `run.bat` or run in terminal:
```powershell
.\run.bat
```
*This starts the FastAPI server and automatically opens your default browser at `http://localhost:8000`.*

#### Option B: Standalone Desktop OpenCV Window
Double-click `desktop.bat` or run:
```powershell
.\desktop.bat
```

---

## ⌨️ Interactive Keyboard Shortcuts

| Key | Context | Action |
|---|---|---|
| **Spacebar** | Word / Sentence Mode | **Next Letter / Save Active Slot** |
| **Backspace** | Word / Sentence Mode | **Previous Letter / Undo Active Slot** |
| **Enter** | Sentence Mode | **Complete Word & Advance to Next Word** |
| **C** | All Modes | **Clear / Reset Active Slots or History** |
| **Escape** | UI | **Close ISL Reference Guide Modal** |

---

## 📁 Repository Structure

```
Sign Language Interpreter/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI server & REST endpoints
│   └── model_service.py           # MediaPipe hand isolation, 179-D feature extractor, ensemble inference
├── static/
│   ├── index.html                 # Modern responsive glassmorphic UI
│   ├── style.css                  # Custom dark-theme glass styling
│   ├── app.js                     # Camera capture, canvas skeleton overlay, mode state machines, TTS
│   └── placeholder_hand.png       # Default hand preview asset
├── dataset_ISL/                   # 26 ISL alphabet classes (12,636 images)
├── ISL_Dataset/                   # Multi-signer ISL evaluation corpus
├── hand_landmarker.task           # Official Google MediaPipe 3D Hand Landmarker model
├── isl_landmark_model.joblib      # 97.57% accuracy ExtraTrees + MLP ensemble
├── train_perfect_model.py         # Complete training pipeline with dual-padding landmark extraction
├── run_interpreter.py             # Web application launcher
├── desktop_interpreter.py         # Standalone desktop OpenCV interface
├── install_dependencies.bat       # Windows dependency installer
├── run.bat                        # Web launcher batch script
├── desktop.bat                    # Desktop launcher batch script
├── requirements.txt               # Complete Python package requirements
├── INTERNATIONAL_FORUM_GUIDE.md   # Comprehensive technical dossier & international presentation script
└── README.md                      # Primary project overview
```

---

## 🔬 Tech Stack Summary

- **Vision & Kinematics**: Google MediaPipe LiteRT (`tasks-vision`), OpenCV (`cv2`)
- **Machine Learning & Numerics**: Scikit-Learn (`ExtraTreesClassifier`, `MLPClassifier`, `VotingClassifier`), NumPy, SciPy, Joblib
- **Backend & Network**: FastAPI, Uvicorn (ASGI event loop), Starlette, Pydantic
- **Frontend & Audio**: HTML5 Canvas, Vanilla ES6+ Asynchronous JavaScript, CSS3 Glassmorphism, W3C Web Speech API (`SpeechSynthesis`)
