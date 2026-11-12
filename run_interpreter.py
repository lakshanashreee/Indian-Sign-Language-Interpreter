import os
import sys
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.5)
    url = "http://localhost:8000"
    print(f"\n[Launcher] Launching Indian Sign Language Interpreter at: {url}")
    webbrowser.open(url)

def main():
    print("=" * 65)
    print("      INDIAN SIGN LANGUAGE (ISL) REAL-TIME INTERPRETER")
    print("=" * 65)
    print("  * Alphabet Mode: Continuous Sign Recognition")
    print("  * Word Mode: Step-by-Step Letter Slot Construction")
    print("  * Sentence Mode: Multi-Word Translation & Speech")
    print("  * Hand-Body Separation: MediaPipe 3D Landmark Isolation")
    print("=" * 65)

    # Check model file
    if not os.path.exists("isl_landmark_model.joblib"):
        print("\n[Notice] Trained model 'isl_landmark_model.joblib' not found!")
        print("[Notice] Running train_landmarks.py to train the model first...")
        import subprocess
        subprocess.run([sys.executable, "-u", "train_landmarks.py"])

    # Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    print("\n[Server] Starting FastAPI Uvicorn Server on http://127.0.0.1:8000 ...")
    print("[Server] Press Ctrl+C in this terminal to stop.\n")
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()

