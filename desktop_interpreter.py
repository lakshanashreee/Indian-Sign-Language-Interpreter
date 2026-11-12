import os
import sys
import time
import cv2
import numpy as np
from backend.model_service import ISLModelService

def speak_text(text):
    """Speaks text using Windows PowerShell SpeechSynthesizer."""
    if not text or text == "-": return
    import subprocess
    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception:
        pass

def main():
    print("=" * 60)
    print("   ISL DESKTOP INTERPRETER (Landmark Neural Mode)")
    print("=" * 60)
    print("Controls:")
    print("  [1] Alphabet Mode")
    print("  [2] Word Mode")
    print("  [3] Sentence Mode")
    print("  [SPACE] Next Letter / Save Slot (Word & Sentence)")
    print("  [ENTER] Complete Word (Sentence Mode)")
    print("  [C] Clear / Reset Current Mode")
    print("  [O] Toggle Camera ON / OFF")
    print("  [Q] Quit")
    print("=" * 60)

    service = ISLModelService()
    
    cap = cv2.VideoCapture(0)
    cam_active = True
    if not cap.isOpened():
        print("[Warning] Camera index 0 could not be opened directly.")
        cam_active = False

    mode = "alphabet"
    
    # Word Mode state
    word_target_len = 4
    word_slots = ["_"] * word_target_len
    word_active_slot = 0
    assembled_word = ""

    # Sentence Mode state
    sentence_target_words = 3
    completed_words = []
    current_word_letters = []
    full_sentence = ""

    prev_time = time.time()
    fps = 0

    while True:
        if cam_active and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Frame Received", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA IS OFF (Press 'O' to Turn On)", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        fps = 1.0 / max(0.001, now - prev_time)
        prev_time = now

        pred_letter = "-"
        conf = 0.0
        hand_detected = False

        if cam_active:
            res = service.predict_sign(frame, use_roi_fallback=False)
            hand_detected = res["hand_detected"]
            pred_letter = res["smoothed_letter"] or "-"
            conf = res["confidence"]
            
            # Draw landmarks
            if res["landmarks"]:
                for hand in res["landmarks"]:
                    for lm in hand:
                        px = int(lm["x"] * w)
                        py = int(lm["y"] * h)
                        cv2.circle(frame, (px, py), 4, (0, 255, 128), -1)
            
            # Draw bounding box
            if res["bbox"] and res["bbox"]["w"] > 0 and hand_detected:
                bb = res["bbox"]
                cv2.rectangle(frame, (bb["x1"], bb["y1"]), (bb["x2"], bb["y2"]), (0, 255, 128), 2)
                cv2.putText(frame, "HAND DETECTED (FACE FILTERED)", (bb["x1"], max(20, bb["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        # Draw Top HUD
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 24, 30), -1)
        mode_str = f"MODE: {mode.upper()}  |  FPS: {int(fps)}"
        cv2.putText(frame, mode_str, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw Live Letter Badge
        cv2.rectangle(frame, (w - 200, 5), (w - 10, 50), (40, 48, 60), -1)
        display_sign = pred_letter if hand_detected else "NO HAND"
        sign_color = (0, 255, 128) if hand_detected and pred_letter != "-" else (120, 120, 120)
        cv2.putText(frame, f"SIGN: {display_sign}", (w - 190, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sign_color, 2)

        # Mode Specific Overlays
        if mode == "alphabet":
            cv2.rectangle(frame, (0, h - 70), (w, h), (20, 24, 30), -1)
            if hand_detected and pred_letter != "-":
                txt = f"Recognized Letter: {pred_letter} ({int(conf*100)}% Conf)"
            else:
                txt = "No Hands in Frame — Show your hand(s) to sign"
            cv2.putText(frame, txt, (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        elif mode == "word":
            cv2.rectangle(frame, (0, h - 90), (w, h), (20, 24, 30), -1)
            slot_str = "Slots: " + " ".join([f"[{c}]" if idx != word_active_slot else f"*[ {c} ]*" for idx, c in enumerate(word_slots)])
            cv2.putText(frame, slot_str, (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 200), 2)
            cv2.putText(frame, f"Slot #{word_active_slot+1} -> [SPACE]: Save Letter | [C]: Reset", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if assembled_word:
                cv2.putText(frame, f"Word: {assembled_word}", (w - 220, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        elif mode == "sentence":
            cv2.rectangle(frame, (0, h - 90), (w, h), (20, 24, 30), -1)
            curr_spelled = "".join(current_word_letters) or "..."
            cv2.putText(frame, f"Word {len(completed_words)+1}/{sentence_target_words}: [{curr_spelled}] (Space: Add, Enter: Next Word)", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            sentence_txt = "Sentence: " + (" ".join(completed_words) or "...")
            cv2.putText(frame, sentence_txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Indian Sign Language Interpreter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            cam_active = not cam_active
            if cam_active and (cap is None or not cap.isOpened()):
                cap = cv2.VideoCapture(0)
            elif not cam_active and cap is not None:
                cap.release()
                cap = None
        elif key == ord('1'):
            mode = "alphabet"
        elif key == ord('2'):
            mode = "word"
            word_slots = ["_"] * word_target_len
            word_active_slot = 0
            assembled_word = ""
        elif key == ord('3'):
            mode = "sentence"
            completed_words = []
            current_word_letters = []
            full_sentence = ""
        elif key == ord('c'):
            if mode == "word":
                word_slots = ["_"] * word_target_len
                word_active_slot = 0
                assembled_word = ""
            elif mode == "sentence":
                completed_words = []
                current_word_letters = []
                full_sentence = ""
        elif key == 32:  # Spacebar
            if hand_detected and pred_letter != "-":
                if mode == "word":
                    if word_active_slot < word_target_len:
                        word_slots[word_active_slot] = pred_letter
                        word_active_slot += 1
                        if word_active_slot >= word_target_len:
                            assembled_word = "".join(word_slots)
                            speak_text(assembled_word)
                elif mode == "sentence":
                    current_word_letters.append(pred_letter)
        elif key == 13:  # Enter
            if mode == "sentence":
                if current_word_letters:
                    word = "".join(current_word_letters)
                    completed_words.append(word)
                    current_word_letters = []
                    if len(completed_words) >= sentence_target_words:
                        full_sentence = " ".join(completed_words)
                        speak_text(full_sentence)

    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
