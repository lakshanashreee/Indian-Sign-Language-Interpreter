import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.model_service import ISLModelService

app = FastAPI(
    title="Indian Sign Language (ISL) Interpreter",
    description="High-Accuracy Real-Time ISL Alphabet, Word, and Sentence Interpreter",
    version="2.0.0"
)

# Enable CORS for browser integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Service
isl_service = ISLModelService()

class PredictRequest(BaseModel):
    image: str  # Base64 encoded JPEG or PNG
    use_fallback: Optional[bool] = True

# ISL Alphabet Guide Descriptions
ISL_ALPHABET_GUIDE = {
    "A": "Two closed fists placed side by side with knuckles facing outward or thumb upright touching the palm.",
    "B": "Both open flat palms touching together along the sides (like a book closing).",
    "C": "Dominant hand forms a curved 'C' shape with thumb and fingers separated.",
    "D": "Dominant index finger pointed straight up, while thumb and fingers form a loop or second index touches tip.",
    "E": "Dominant index finger resting on the palm or fingers curled inward against the thumb.",
    "F": "Index finger and thumb form an open pinch/circle with other three fingers extended upward.",
    "G": "Both fists held horizontally or index and thumb pointing parallel forward.",
    "H": "Dominant flat horizontal hand sweeping or resting on the non-dominant palm.",
    "I": "Pinky finger held vertically straight up from a closed fist.",
    "J": "Pinky finger extended and traces the curved hook shape of the letter 'J' in the air.",
    "K": "Index finger pointed up with middle finger angled out, thumb tucked between them.",
    "L": "Index finger pointing up and thumb extended out at 90 degrees forming an 'L' shape.",
    "M": "Three fingers (index, middle, ring) draped over the thumb in a closed fist or three fingers extended downward.",
    "N": "Two fingers (index, middle) draped over the thumb in a closed fist.",
    "O": "All fingertips and thumb curled to touch, forming an 'O' circle shape.",
    "P": "Index pointing downward with middle finger extended outward, thumb supporting.",
    "Q": "Fingers pointing downward in a circle or hook with thumb touching index.",
    "R": "Index and middle fingers crossed tightly over each other.",
    "S": "Closed fist with thumb wrapped across the front of the curled fingers.",
    "T": "Thumb tucked between the index and middle fingers in a closed fist or index placed across opposite index.",
    "U": "Index and middle fingers held straight up and glued closely together.",
    "V": "Index and middle fingers held straight up spread apart forming a 'V' shape.",
    "W": "Three middle fingers (index, middle, ring) held straight up spread apart forming a 'W' shape.",
    "X": "Index finger curled into a hook shape from a closed fist.",
    "Y": "Thumb and pinky extended wide outward ('hang loose' gesture) with middle fingers folded.",
    "Z": "Index finger traces the zigzag 'Z' pattern in the air or index placed across the opposite palm."
}

@app.get("/api/info")
def get_info():
    return {
        "model_name": "ISL Multi-Layer Perceptron (HOG + Hand Isolation)",
        "classes": isl_service.classes,
        "num_classes": len(isl_service.classes),
        "accuracy": round(isl_service.accuracy * 100, 2),
        "detector_available": isl_service.detector is not None,
        "features": [
            "Hand Isolation from Face & Body via MediaPipe",
            "Real-Time Alphabet Mode with Confidence Gauging",
            "Interactive Word Mode with Letter Slot Progression",
            "Full Sentence Mode with Multi-Word Assembly",
            "Text-to-Speech Pronunciation",
            "Visual Hand Crop Preview"
        ]
    }

@app.get("/api/reference")
def get_reference():
    return JSONResponse(content=ISL_ALPHABET_GUIDE)

@app.post("/api/reset_history")
def reset_history():
    isl_service.history.clear()
    return {"status": "ok", "message": "History cleared"}

@app.post("/api/predict")
def predict_frame(req: PredictRequest):
    try:
        # Decode base64 image
        img_str = req.image
        if "," in img_str:
            img_str = img_str.split(",")[1]
            
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        result = isl_service.predict_sign(frame, use_roi_fallback=req.use_fallback)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ISL Interpreter API is running. index.html not yet created."}

