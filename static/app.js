/**
 * Indian Sign Language (ISL) Interpreter
 * Real-Time Vision & Landmark Interaction Engine
 */

// Application State
const state = {
  cameraActive: false,
  isMirrored: true,
  showSkeleton: true,
  showFramingBox: false, // Default off to reduce visual clutter
  enableTTS: true,
  activeMode: 'alphabet', // 'alphabet' | 'word' | 'sentence'
  stream: null,
  processingLoop: null,
  fpsCounter: 0,
  lastFpsTimestamp: performance.now(),
  currentFps: 0,

  // Real-time Recognition Output
  currentLetter: '-',
  currentSmoothedLetter: '-',
  currentConfidence: 0.0,
  top3: [],
  handDetected: false,
  handsCount: 0,
  lastHandCrop: null,

  // Alphabet Mode State
  alphaHistory: [],
  lastSpokenAlpha: null,

  // Word Mode State
  wordTargetLength: 4,
  wordSlots: [],
  activeWordSlotIndex: 0,
  assembledWord: '',

  // Sentence Mode State
  sentenceTargetWords: 3,
  currentWordIndex: 0,
  completedWords: [],
  currentWordLetters: [],
  fullSentence: '',

  // System Guide Cache
  guideData: null
};

// MediaPipe standard 21-hand landmark connections
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  // Palm Base
  [0, 17]
];

// DOM Elements
const videoEl = document.getElementById('videoElement');
const canvasEl = document.getElementById('overlayCanvas');
const ctx = canvasEl.getContext('2d');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const framingBox = document.getElementById('framingBox');
const btnToggleCamera = document.getElementById('btnToggleCamera');
const btnPlaceholderStartCam = document.getElementById('btnPlaceholderStartCam');
const camBtnIcon = document.getElementById('camBtnIcon');
const camBtnText = document.getElementById('camBtnText');
const btnToggleSkeleton = document.getElementById('btnToggleSkeleton');
const btnToggleBox = document.getElementById('btnToggleBox');
const btnFlipCam = document.getElementById('btnFlipCam');
const detectionBadge = document.getElementById('detectionBadge');
const fpsBadge = document.getElementById('fpsBadge');
const isolationState = document.getElementById('isolationState');
const cropPreviewImg = document.getElementById('cropPreviewImg');
const handsCountVal = document.getElementById('handsCountVal');
const trackingStatus = document.getElementById('trackingStatus');
const heroLetter = document.getElementById('heroLetter');
const letterName = document.getElementById('letterName');
const confBadge = document.getElementById('confBadge');
const confidenceBar = document.getElementById('confidenceBar');
const top3Row = document.getElementById('top3Row');
const btnToggleTTS = document.getElementById('btnToggleTTS');
const ttsIcon = document.getElementById('ttsIcon');
const ttsText = document.getElementById('ttsText');

// Mode DOM
const modeTabs = document.querySelectorAll('.mode-tab');
const paneAlphabet = document.getElementById('paneAlphabet');
const paneWord = document.getElementById('paneWord');
const paneSentence = document.getElementById('paneSentence');

// Alphabet Mode DOM
const bigAlphaDisplay = document.getElementById('bigAlphaDisplay');
const bigAlphaDesc = document.getElementById('bigAlphaDesc');
const btnSpeakAlpha = document.getElementById('btnSpeakAlpha');
const btnCopyAlpha = document.getElementById('btnCopyAlpha');
const btnClearAlphaHistory = document.getElementById('btnClearAlphaHistory');
const alphaHistoryContainer = document.getElementById('alphaHistoryContainer');

// Word Mode DOM
const numLettersInput = document.getElementById('numLettersInput');
const btnInitWord = document.getElementById('btnInitWord');
const wordSlotsSection = document.getElementById('wordSlotsSection');
const wordSlotsContainer = document.getElementById('wordSlotsContainer');
const activeSlotIndicator = document.getElementById('activeSlotIndicator');
const slotInterpretingVal = document.getElementById('slotInterpretingVal');
const slotInterpretingConf = document.getElementById('slotInterpretingConf');
const btnNextLetter = document.getElementById('btnNextLetter');
const btnPrevLetter = document.getElementById('btnPrevLetter');
const btnResetWord = document.getElementById('btnResetWord');
const wordResultCard = document.getElementById('wordResultCard');
const assembledWordDisplay = document.getElementById('assembledWordDisplay');
const btnSpeakWord = document.getElementById('btnSpeakWord');
const btnCopyWord = document.getElementById('btnCopyWord');
const btnAddWordToSentence = document.getElementById('btnAddWordToSentence');
const btnNewWord = document.getElementById('btnNewWord');

// Sentence Mode DOM
const sentenceConfigBox = document.getElementById('sentenceConfigBox');
const numWordsInput = document.getElementById('numWordsInput');
const btnInitSentence = document.getElementById('btnInitSentence');
const sentenceWorkspace = document.getElementById('sentenceWorkspace');
const wordsBreadcrumbBar = document.getElementById('wordsBreadcrumbBar');
const activeWordTitle = document.getElementById('activeWordTitle');
const wordLetterCountBadge = document.getElementById('wordLetterCountBadge');
const sentenceLetterSlots = document.getElementById('sentenceLetterSlots');
const sentenceInterpretingVal = document.getElementById('sentenceInterpretingVal');
const btnSentenceNextLetter = document.getElementById('btnSentenceNextLetter');
const btnSentenceBackLetter = document.getElementById('btnSentenceBackLetter');
const btnCompleteWordNext = document.getElementById('btnCompleteWordNext');
const fullSentenceDisplay = document.getElementById('fullSentenceDisplay');
const sentenceCharCount = document.getElementById('sentenceCharCount');
const btnSpeakSentence = document.getElementById('btnSpeakSentence');
const btnCopySentence = document.getElementById('btnCopySentence');
const btnResetSentence = document.getElementById('btnResetSentence');

// Modal DOM
const guideModal = document.getElementById('guideModal');
const btnOpenGuide = document.getElementById('btnOpenGuide');
const btnCloseGuide = document.getElementById('btnCloseGuide');
const btnCloseGuideBtn = document.getElementById('btnCloseGuideBtn');
const guideGrid = document.getElementById('guideGrid');

/* ==========================================================
   INITIALIZATION & EVENT LISTENERS
   ========================================================== */
window.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  fetchSystemInfo();
  initWordMode(4);
  if (framingBox) framingBox.style.display = 'none';
});

function initEventListeners() {
  btnToggleCamera.addEventListener('click', toggleCamera);
  btnPlaceholderStartCam.addEventListener('click', toggleCamera);

  btnToggleSkeleton.addEventListener('click', () => {
    state.showSkeleton = !state.showSkeleton;
    btnToggleSkeleton.innerHTML = `<span>🦴</span> Skeleton: ${state.showSkeleton ? 'ON' : 'OFF'}`;
  });

  btnToggleBox.addEventListener('click', () => {
    state.showFramingBox = !state.showFramingBox;
    if (framingBox) framingBox.style.display = state.showFramingBox ? 'block' : 'none';
    btnToggleBox.innerHTML = `<span>🎯</span> Guide Box: ${state.showFramingBox ? 'ON' : 'OFF'}`;
  });

  btnFlipCam.addEventListener('click', () => {
    state.isMirrored = !state.isMirrored;
    videoEl.style.transform = state.isMirrored ? 'scaleX(-1)' : 'scaleX(1)';
  });

  btnToggleTTS.addEventListener('click', () => {
    state.enableTTS = !state.enableTTS;
    ttsIcon.textContent = state.enableTTS ? '🔊' : '🔇';
    ttsText.textContent = `Voice: ${state.enableTTS ? 'ON' : 'OFF'}`;
  });

  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      switchMode(tab.dataset.mode);
    });
  });

  // Alphabet Mode Actions
  btnSpeakAlpha.addEventListener('click', () => {
    if (state.currentSmoothedLetter && state.currentSmoothedLetter !== '-') {
      speakText(`Alphabet ${state.currentSmoothedLetter}`);
    }
  });
  btnCopyAlpha.addEventListener('click', () => {
    if (state.currentSmoothedLetter && state.currentSmoothedLetter !== '-') {
      navigator.clipboard.writeText(state.currentSmoothedLetter);
      showToast(`Copied "${state.currentSmoothedLetter}" to clipboard!`);
    }
  });
  btnClearAlphaHistory.addEventListener('click', () => {
    state.alphaHistory = [];
    renderAlphaHistory();
    fetch('/api/reset_history', { method: 'POST' }).catch(() => { });
  });

  // Word Mode Actions
  btnInitWord.addEventListener('click', () => {
    const val = parseInt(numLettersInput.value) || 4;
    initWordMode(val);
  });
  document.querySelectorAll('.preset-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const len = parseInt(pill.dataset.len);
      numLettersInput.value = len;
      initWordMode(len);
    });
  });
  btnNextLetter.addEventListener('click', saveActiveWordSlot);
  btnPrevLetter.addEventListener('click', prevActiveWordSlot);
  btnResetWord.addEventListener('click', () => initWordMode(state.wordTargetLength));
  btnSpeakWord.addEventListener('click', () => speakText(state.assembledWord));
  btnCopyWord.addEventListener('click', () => {
    navigator.clipboard.writeText(state.assembledWord);
    showToast(`Copied "${state.assembledWord}"!`);
  });
  btnAddWordToSentence.addEventListener('click', () => {
    switchMode('sentence');
      if (!sentenceWorkspace.style.display || sentenceWorkspace.style.display === 'none') {
        initSentenceMode(3);
      }
      state.completedWords.push(state.assembledWord);
      updateSentenceOutput();
      showToast(`Word "${state.assembledWord}" added to sentence!`);
    });
  btnNewWord.addEventListener('click', () => initWordMode(state.wordTargetLength));

  // Sentence Mode Actions
  btnInitSentence.addEventListener('click', () => {
    const val = parseInt(numWordsInput.value) || 3;
    initSentenceMode(val);
  });
  document.querySelectorAll('.preset-pill-sentence').forEach(pill => {
    pill.addEventListener('click', () => {
      const w = parseInt(pill.dataset.words);
      numWordsInput.value = w;
      initSentenceMode(w);
    });
  });
  btnSentenceNextLetter.addEventListener('click', saveSentenceLetter);
  btnSentenceBackLetter.addEventListener('click', backSentenceLetter);
  btnCompleteWordNext.addEventListener('click', completeCurrentWordAndAdvance);
  btnSpeakSentence.addEventListener('click', () => speakText(state.fullSentence));
  btnCopySentence.addEventListener('click', () => {
    navigator.clipboard.writeText(state.fullSentence);
    showToast('Sentence copied to clipboard!');
  });
  btnResetSentence.addEventListener('click', () => initSentenceMode(state.sentenceTargetWords));

  // Guide Modal
  btnOpenGuide.addEventListener('click', openGuideModal);
  btnCloseGuide.addEventListener('click', closeGuideModal);
  btnCloseGuideBtn.addEventListener('click', closeGuideModal);
  guideModal.addEventListener('click', (e) => {
    if (e.target === guideModal) closeGuideModal();
  });

  window.addEventListener('keydown', handleKeyNavigation);
}

/* ==========================================================
   CAMERA & VISION ENGINE
   ========================================================== */
async function toggleCamera() {
  if (state.cameraActive) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    };

    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = state.stream;

    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;

    state.cameraActive = true;
    cameraPlaceholder.style.display = 'none';
    camBtnIcon.textContent = '⏹️';
    camBtnText.textContent = 'Turn Off Camera';
    btnToggleCamera.classList.add('btn-accent');
    btnToggleCamera.classList.remove('btn-primary');
    detectionBadge.className = 'badge badge-success';
    detectionBadge.textContent = '🟢 Camera Running';

    startProcessingLoop();
  } catch (err) {
    console.error('Camera access failed:', err);
    alert('Unable to access camera. Please allow camera permissions in your browser.');
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }
  if (state.processingLoop) {
    clearInterval(state.processingLoop);
    state.processingLoop = null;
  }

  state.cameraActive = false;
  cameraPlaceholder.style.display = 'flex';
  camBtnIcon.textContent = '▶️';
  camBtnText.textContent = 'Turn On Camera';
  btnToggleCamera.classList.remove('btn-accent');
  btnToggleCamera.classList.add('btn-primary');
  detectionBadge.className = 'badge';
  detectionBadge.textContent = '⚪ Camera Off';
  fpsBadge.textContent = '0 FPS';

  handleInferenceResult({
    hand_detected: false,
    letter: null,
    smoothed_letter: '-',
    confidence: 0.0,
    top3: [],
    landmarks: [],
    bbox: null,
    hand_count: 0,
    hand_crop_base64: null
  });

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

function startProcessingLoop() {
  const offscreenCanvas = document.createElement('canvas');
  const offscreenCtx = offscreenCanvas.getContext('2d');

  let isRequestPending = false;

  state.processingLoop = setInterval(async () => {
    if (!state.cameraActive || isRequestPending) return;

    state.fpsCounter++;
    const now = performance.now();
    if (now - state.lastFpsTimestamp >= 1000) {
      state.currentFps = state.fpsCounter;
      fpsBadge.textContent = `${state.currentFps} FPS`;
      state.fpsCounter = 0;
      state.lastFpsTimestamp = now;
    }

    offscreenCanvas.width = videoEl.videoWidth || 640;
    offscreenCanvas.height = videoEl.videoHeight || 480;
    offscreenCtx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const base64Data = offscreenCanvas.toDataURL('image/jpeg', 0.65);

    isRequestPending = true;
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data, use_fallback: false })
      });

      if (resp.ok) {
        const data = await resp.json();
        handleInferenceResult(data);
      }
    } catch (err) {
      console.warn('Inference error:', err);
    } finally {
      isRequestPending = false;
    }
}, 80);
}

function handleInferenceResult(data) {
  state.handDetected = data.hand_detected;
  state.currentLetter = data.letter || '-';
  state.currentSmoothedLetter = data.smoothed_letter || '-';
  state.currentConfidence = data.confidence || 0.0;
  state.top3 = data.top3 || [];
  state.handsCount = data.hand_count || 0;

  // Update Isolated Hand Crop Thumbnail
  if (data.hand_crop_base64) {
    cropPreviewImg.src = data.hand_crop_base64;
    state.lastHandCrop = data.hand_crop_base64;
  }

  // Update Status Badges
  handsCountVal.textContent = state.handsCount;

  if (state.handDetected) {
    isolationState.className = 'badge badge-success';
    isolationState.textContent = 'Body & Face Filtered';
    trackingStatus.textContent = `${state.handsCount} Hand(s) Locked`;
    detectionBadge.className = 'badge badge-success';
    detectionBadge.textContent = `🟢 Hands Active (${state.handsCount})`;
  } else {
    isolationState.className = 'badge';
    isolationState.textContent = 'Idle (No Hands)';
    trackingStatus.textContent = 'Show hands to camera';
    detectionBadge.className = 'badge';
    detectionBadge.textContent = '⚪ No Hands in Frame';
  }

    // Update Prediction Hero Box
    if (state.handDetected && state.currentSmoothedLetter !== '-') {
      heroLetter.textContent = state.currentSmoothedLetter;
      const pct = Math.round(state.currentConfidence * 100);
      confBadge.textContent = `${pct}% Conf`;
      confidenceBar.style.width = `${pct}%`;
      letterName.textContent = `Sign Recognized: ${state.currentSmoothedLetter}`;
    } else {
      heroLetter.textContent = '-';
      confBadge.textContent = '0% Conf';
      confidenceBar.style.width = '0%';
      letterName.textContent = state.cameraActive
        ? 'Awaiting Hands in Frame...'
        : 'Camera is Off';
    }

    // Update Top-3 Pills
    top3Row.innerHTML = '';
    if (state.handDetected && state.top3.length > 0) {
      state.top3.forEach(item => {
        const pill = document.createElement('span');
        pill.className = 'top3-pill';
        pill.textContent = `${item.letter}: ${Math.round(item.prob * 100)}%`;
        top3Row.appendChild(pill);
      });
    } else {
      top3Row.innerHTML = '<span class="top3-pill">-</span><span class="top3-pill">-</span><span class="top3-pill">-</span>';
    }

    // Draw overlay skeleton on canvas
    drawHandLandmarks(data.landmarks, data.bbox);

    // Dispatch to active mode
    dispatchLiveRecognition(state.currentSmoothedLetter, state.currentConfidence, state.handDetected);
  }

  function drawHandLandmarks(landmarksList, bbox) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (!state.showSkeleton) return;

    const w = canvasEl.width;
    const h = canvasEl.height;

    // Draw bounding box around isolated hands
      if (bbox && bbox.w > 0 && state.handDetected) {
        ctx.save();
        let bx1 = bbox.x1;
        let bx2 = bbox.x2;
        if (state.isMirrored) {
          const temp1 = w - bx2;
          const temp2 = w - bx1;
          bx1 = temp1;
          bx2 = temp2;
        }
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.strokeRect(bx1, bbox.y1, bx2 - bx1, bbox.h);

        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
        ctx.fillText('ISOLATED HAND (FACE FILTERED)', bx1 + 6, Math.max(16, bbox.y1 - 6));
        ctx.restore();
      }

      // Draw skeletal landmarks
      if (landmarksList && landmarksList.length > 0) {
        landmarksList.forEach((hand) => {
          // Draw Connections (Bones)
          ctx.strokeStyle = '#6366f1';
          ctx.lineWidth = 3;
          HAND_CONNECTIONS.forEach(([i, j]) => {
            if (hand[i] && hand[j]) {
              let x1 = hand[i].x * w;
              let y1 = hand[i].y * h;
              let x2 = hand[j].x * w;
              let y2 = hand[j].y * h;

              if (state.isMirrored) {
                x1 = w - x1;
                x2 = w - x2;
              }

              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x2, y2);
              ctx.stroke();
            }
          });

          // Draw Landmark Points (Joints)
          hand.forEach((lm) => {
            let lx = lm.x * w;
            let ly = lm.y * h;
            if (state.isMirrored) {
              lx = w - lx;
            }

            ctx.fillStyle = '#10b981';
            ctx.beginPath();
            ctx.arc(lx, ly, 4.5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          });
        });
      }
    }

    /* ==========================================================
       MODE SWITCHING & DISPATCH
       ========================================================== */
    function switchMode(newMode) {
      state.activeMode = newMode;
      modeTabs.forEach(t => {
        t.classList.toggle('active', t.dataset.mode === newMode);
      });
      paneAlphabet.classList.toggle('active', newMode === 'alphabet');
      paneWord.classList.toggle('active', newMode === 'word');
      paneSentence.classList.toggle('active', newMode === 'sentence');
    }

      function dispatchLiveRecognition(letter, confidence, handDetected) {
        if (state.activeMode === 'alphabet') {
          updateAlphabetMode(letter, confidence, handDetected);
        } else if (state.activeMode === 'word') {
          updateWordModePreview(letter, confidence, handDetected);
        } else if (state.activeMode === 'sentence') {
          updateSentenceModePreview(letter, confidence, handDetected);
        }
      }

      /* ==========================================================
         MODE 1: ALPHABET MODE
         ========================================================== */
        function updateAlphabetMode(letter, confidence, handDetected) {
          if (!handDetected || !letter || letter === '-') {
            bigAlphaDisplay.textContent = '?';
            bigAlphaDesc.textContent = state.cameraActive
              ? 'Perform an ISL sign with your hand in front of the camera.'
              : 'Turn on camera to start interpreting.';
            return;
          }

          bigAlphaDisplay.textContent = letter;

          if (state.guideData && state.guideData[letter]) {
            bigAlphaDesc.textContent = state.guideData[letter];
          } else {
            bigAlphaDesc.textContent = `ISL letter "${letter}" (${Math.round(confidence * 100)}% confidence)`;
          }

            // Append to history if stable and confidence >= 0.45
            if (confidence >= 0.45 && letter !== state.lastSpokenAlpha) {
              state.alphaHistory.unshift({ letter, time: new Date().toLocaleTimeString() });
              if (state.alphaHistory.length > 20) state.alphaHistory.pop();
              renderAlphaHistory();

              if (state.enableTTS) {
                speakText(`Letter ${letter}`);
              }
              state.lastSpokenAlpha = letter;
            }
          }

          function renderAlphaHistory() {
            if (state.alphaHistory.length === 0) {
              alphaHistoryContainer.innerHTML = '<span class="history-empty">Signs will appear here as you perform them...</span>';
              return;
            }

            alphaHistoryContainer.innerHTML = '';
            state.alphaHistory.forEach(item => {
              const chip = document.createElement('span');
              chip.className = 'history-chip';
              chip.textContent = item.letter;
              alphaHistoryContainer.appendChild(chip);
            });
          }

          /* ==========================================================
             MODE 2: WORD MODE
             ========================================================== */
          function initWordMode(length) {
            state.wordTargetLength = Math.max(1, Math.min(15, length));
            state.activeWordSlotIndex = 0;
            state.wordSlots = [];
            state.assembledWord = '';

            for (let i = 0; i < state.wordTargetLength; i++) {
              state.wordSlots.push({
                index: i + 1,
                char: '',
                locked: false
              });
            }

            wordResultCard.style.display = 'none';
            renderWordSlots();
            updateWordActiveSlotHUD();
          }

          function renderWordSlots() {
            wordSlotsContainer.innerHTML = '';
            state.wordSlots.forEach((slot, idx) => {
              const slotCard = document.createElement('div');
              slotCard.className = 'slot-card';
              if (idx === state.activeWordSlotIndex) {
                slotCard.classList.add('active');
              }
              if (slot.locked) {
                slotCard.classList.add('filled');
              }

              const numEl = document.createElement('div');
              numEl.className = 'slot-num';
              numEl.textContent = `#${slot.index}`;

              const charEl = document.createElement('div');
              charEl.className = 'slot-char';
              charEl.textContent = slot.char || (idx === state.activeWordSlotIndex ? '?' : '_');

              slotCard.appendChild(numEl);
              slotCard.appendChild(charEl);

              slotCard.addEventListener('click', () => {
                state.activeWordSlotIndex = idx;
                renderWordSlots();
                updateWordActiveSlotHUD();
              });

              wordSlotsContainer.appendChild(slotCard);
            });
          }

          function updateWordActiveSlotHUD() {
            const currentSlotNum = state.activeWordSlotIndex + 1;
            activeSlotIndicator.textContent = `Active: Slot ${currentSlotNum} of ${state.wordTargetLength}`;

            if (state.handDetected && state.currentSmoothedLetter !== '-') {
              slotInterpretingVal.textContent = state.currentSmoothedLetter;
              const conf = Math.round(state.currentConfidence * 100);
              slotInterpretingConf.textContent = `(${conf}% conf)`;
            } else {
              slotInterpretingVal.textContent = '-';
              slotInterpretingConf.textContent = '(Awaiting Hand)';
            }
          }

            function updateWordModePreview(letter, confidence, handDetected) {
              if (state.activeWordSlotIndex < state.wordTargetLength) {
                if (handDetected && letter !== '-') {
                  slotInterpretingVal.textContent = letter;
                  const conf = Math.round(confidence * 100);
                  slotInterpretingConf.textContent = `(${conf}% conf)`;

                        const activeSlot = state.wordSlots[state.activeWordSlotIndex];
                        if (activeSlot && !activeSlot.locked) {
                          const activeCard = wordSlotsContainer.children[state.activeWordSlotIndex];
                          if (activeCard) {
                            const charEl = activeCard.querySelector('.slot-char');
                            if (charEl) charEl.textContent = letter;
                          }
                        }
                      } else {
                        slotInterpretingVal.textContent = '-';
                        slotInterpretingConf.textContent = '(Awaiting Hand)';
                        const activeSlot = state.wordSlots[state.activeWordSlotIndex];
                        if (activeSlot && !activeSlot.locked) {
                          const activeCard = wordSlotsContainer.children[state.activeWordSlotIndex];
                          if (activeCard) {
                            const charEl = activeCard.querySelector('.slot-char');
                            if (charEl) charEl.textContent = '?';
                          }
                        }
                      }
                    }
                  }

                  function saveActiveWordSlot() {
                    if (state.activeWordSlotIndex >= state.wordTargetLength) return;

                    if (!state.handDetected || !state.currentSmoothedLetter || state.currentSmoothedLetter === '-') {
                      showToast('Show your hand to the camera to interpret a sign first!');
                      return;
                    }

                    const letterToSave = state.currentSmoothedLetter;
                    state.wordSlots[state.activeWordSlotIndex].char = letterToSave;
                    state.wordSlots[state.activeWordSlotIndex].locked = true;

                    if (state.enableTTS) {
                      speakText(`Slot ${state.activeWordSlotIndex + 1}: ${letterToSave}`);
                    }

                    state.activeWordSlotIndex++;

                    if (state.activeWordSlotIndex >= state.wordTargetLength) {
                      finalizeWord();
                    } else {
                      renderWordSlots();
                      updateWordActiveSlotHUD();
                    }
                  }

                  function prevActiveWordSlot() {
                    if (state.activeWordSlotIndex > 0) {
                      state.activeWordSlotIndex--;
                      state.wordSlots[state.activeWordSlotIndex].locked = false;
                      renderWordSlots();
                      updateWordActiveSlotHUD();
                    }
                  }

                  function finalizeWord() {
                    renderWordSlots();
                    state.assembledWord = state.wordSlots.map(s => s.char).join('');
                    assembledWordDisplay.textContent = state.assembledWord;
                    wordResultCard.style.display = 'flex';
                    activeSlotIndicator.textContent = `Word Completed (${state.wordTargetLength} Letters)`;

                    if (state.enableTTS) {
                      speakText(`Completed Word: ${state.assembledWord}`);
                    }
                  }

                  /* ==========================================================
                     MODE 3: SENTENCE MODE
                     ========================================================== */
                  function initSentenceMode(targetWords) {
                    state.sentenceTargetWords = Math.max(1, Math.min(10, targetWords));
                    state.currentWordIndex = 0;
                    state.completedWords = [];
                    state.currentWordLetters = [];
                    state.fullSentence = '';

                    sentenceConfigBox.style.display = 'block';
                    sentenceWorkspace.style.display = 'flex';

                    renderSentenceBreadcrumbs();
                    renderSentenceWordSlots();
                    updateSentenceOutput();
                  }

                  function renderSentenceBreadcrumbs() {
                    wordsBreadcrumbBar.innerHTML = '';
                    for (let i = 0; i < state.sentenceTargetWords; i++) {
                      const pill = document.createElement('div');
                      pill.className = 'word-breadcrumb-pill';
                      if (i === state.currentWordIndex) {
                        pill.classList.add('active');
                        pill.innerHTML = `<span>✏️</span> Word ${i + 1}`;
                      } else if (i < state.currentWordIndex) {
                        pill.classList.add('done');
                        pill.innerHTML = `<span>✅</span> ${state.completedWords[i] || `Word ${i + 1}`}`;
                      } else {
                        pill.innerHTML = `<span>⏳</span> Word ${i + 1}`;
                      }
                      wordsBreadcrumbBar.appendChild(pill);
                    }
                  }

                  function renderSentenceWordSlots() {
                    activeWordTitle.textContent = `Building Word ${state.currentWordIndex + 1} of ${state.sentenceTargetWords}`;
                    wordLetterCountBadge.textContent = `Letters: ${state.currentWordLetters.length}`;

                    sentenceLetterSlots.innerHTML = '';
                    if (state.currentWordLetters.length === 0) {
                      const emptySlot = document.createElement('div');
                      emptySlot.className = 'slot-card active';
                      emptySlot.innerHTML = '<div class="slot-num">#1</div><div class="slot-char">?</div>';
                      sentenceLetterSlots.appendChild(emptySlot);
                    } else {
                      state.currentWordLetters.forEach((char, idx) => {
                        const slot = document.createElement('div');
                        slot.className = 'slot-card filled';
                        slot.innerHTML = `<div class="slot-num">#${idx + 1}</div><div class="slot-char">${char}</div>`;
                        sentenceLetterSlots.appendChild(slot);
                      });
                      const activeSlot = document.createElement('div');
                      activeSlot.className = 'slot-card active';
                      activeSlot.innerHTML = `<div class="slot-num">#${state.currentWordLetters.length + 1}</div><div class="slot-char">?</div>`;
                      sentenceLetterSlots.appendChild(activeSlot);
                    }
                  }

                        function updateSentenceModePreview(letter, confidence, handDetected) {
                          if (handDetected && letter !== '-') {
                            sentenceInterpretingVal.textContent = letter;
                            const activeCard = sentenceLetterSlots.querySelector('.slot-card.active');
                            if (activeCard) {
                              const charEl = activeCard.querySelector('.slot-char');
                              if (charEl) charEl.textContent = letter;
                            }
                          } else {
                            sentenceInterpretingVal.textContent = '-';
                            const activeCard = sentenceLetterSlots.querySelector('.slot-card.active');
                            if (activeCard) {
                              const charEl = activeCard.querySelector('.slot-char');
                              if (charEl) charEl.textContent = '?';
                            }
                          }
                        }

                        function saveSentenceLetter() {
                          if (!state.handDetected || !state.currentSmoothedLetter || state.currentSmoothedLetter === '-') {
                            showToast('Show your hand to the camera to interpret a sign first!');
                            return;
                          }

                          const letterToSave = state.currentSmoothedLetter;
                          state.currentWordLetters.push(letterToSave);
                          if (state.enableTTS) {
                            speakText(letterToSave);
                          }
                          renderSentenceWordSlots();
                        }

                        function backSentenceLetter() {
                          if (state.currentWordLetters.length > 0) {
                            state.currentWordLetters.pop();
                            renderSentenceWordSlots();
                          }
                        }

                        function completeCurrentWordAndAdvance() {
                          if (state.currentWordLetters.length === 0) {
                            showToast('Add at least one letter to this word before advancing.');
                            return;
                          }

                          const word = state.currentWordLetters.join('');
                          state.completedWords.push(word);
                          state.currentWordLetters = [];
                          state.currentWordIndex++;

                          if (state.enableTTS) {
                            speakText(`Word: ${word}`);
                          }

                          updateSentenceOutput();

                          if (state.currentWordIndex >= state.sentenceTargetWords) {
                            renderSentenceBreadcrumbs();
                            activeWordTitle.textContent = '🎉 All Words Completed!';
                            sentenceLetterSlots.innerHTML = '';
                            if (state.enableTTS) {
                              setTimeout(() => speakText(`Complete sentence: ${state.fullSentence}`), 800);
                            }
                          } else {
                            renderSentenceBreadcrumbs();
                            renderSentenceWordSlots();
                          }
                        }

                        function updateSentenceOutput() {
                          state.fullSentence = state.completedWords.join(' ');
                          if (state.fullSentence) {
                            fullSentenceDisplay.textContent = state.fullSentence;
                            sentenceCharCount.textContent = `${state.fullSentence.length} characters (${state.completedWords.length} words)`;
                          } else {
                            fullSentenceDisplay.textContent = 'Your sentence will appear here as words are completed...';
                            sentenceCharCount.textContent = '0 characters';
                          }
                        }

                        /* ==========================================================
                           TEXT TO SPEECH (TTS) & AUDIO
                           ========================================================== */
                        function speakText(text) {
                          if (!state.enableTTS || !text) return;

                          if ('speechSynthesis' in window) {
                            window.speechSynthesis.cancel();
                            const utterance = new SpeechSynthesisUtterance(text);
                            utterance.rate = 1.0;
                            utterance.pitch = 1.0;
                            utterance.lang = 'en-IN';
                            window.speechSynthesis.speak(utterance);
                          }
                        }

                        /* ==========================================================
                           KEYBOARD SHORTCUTS
                           ========================================================== */
                        function handleKeyNavigation(e) {
                          if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                          if (e.code === 'Space') {
                            e.preventDefault();
                            if (state.activeMode === 'word') {
                              saveActiveWordSlot();
                            } else if (state.activeMode === 'sentence') {
                              saveSentenceLetter();
                            }
                          } else if (e.code === 'Backspace') {
                            if (state.activeMode === 'word') {
                              e.preventDefault();
                              prevActiveWordSlot();
                            } else if (state.activeMode === 'sentence') {
                              e.preventDefault();
                              backSentenceLetter();
                            }
                          } else if (e.code === 'Enter') {
                            if (state.activeMode === 'sentence') {
                              e.preventDefault();
                              completeCurrentWordAndAdvance();
                            }
                          } else if (e.key.toLowerCase() === 'c' && !e.ctrlKey) {
                            if (state.activeMode === 'alphabet') {
                              state.alphaHistory = [];
                              renderAlphaHistory();
                            }
                          } else if (e.key === 'Escape') {
                            closeGuideModal();
                          }
                        }

                        /* ==========================================================
                           SYSTEM INFO & ISL GUIDE MODAL
                           ========================================================== */
                        async function fetchSystemInfo() {
                          try {
                            const res = await fetch('/api/info');
                            if (res.ok) {
                              const info = await res.json();
                              const modelStatusText = document.getElementById('modelStatusText');
                              if (modelStatusText) {
                                modelStatusText.textContent = `Landmark Model Ready (26 Classes &bull; ${info.accuracy}% Acc)`;
                              }
                            }
                          } catch (err) {
                            console.warn('Info endpoint unavailable:', err);
                          }

                          try {
                            const res2 = await fetch('/api/reference');
                            if (res2.ok) {
                              state.guideData = await res2.json();
                              renderGuideModal();
                            }
                          } catch (err) {
                            console.warn('Guide endpoint unavailable:', err);
                          }
                        }

                        function openGuideModal() {
                          guideModal.classList.add('open');
                        }

                        function closeGuideModal() {
                          guideModal.classList.remove('open');
                        }

                        function renderGuideModal() {
                          if (!state.guideData) return;
                          guideGrid.innerHTML = '';

                          Object.keys(state.guideData).sort().forEach(letter => {
                            const desc = state.guideData[letter];
                            const card = document.createElement('div');
                            card.className = 'guide-card';
                            card.innerHTML = `
      <div class="guide-letter">${letter}</div>
      <div class="guide-desc">${desc}</div>
    `;
                            guideGrid.appendChild(card);
                          });
                        }

                        function showToast(msg) {
                          const toast = document.createElement('div');
                          toast.style.position = 'fixed';
                          toast.style.bottom = '24px';
                          toast.style.right = '24px';
                          toast.style.background = '#10b981';
                          toast.style.color = '#fff';
                          toast.style.padding = '10px 18px';
                          toast.style.borderRadius = '8px';
                          toast.style.fontWeight = '600';
                          toast.style.fontSize = '0.9rem';
                          toast.style.boxShadow = '0 8px 24px rgba(0,0,0,0.4)';
                          toast.style.zIndex = '9999';
                          toast.style.transition = 'opacity 0.3s ease';
                          toast.textContent = msg;

                          document.body.appendChild(toast);
                          setTimeout(() => {
                            toast.style.opacity = '0';
                            setTimeout(() => toast.remove(), 300);
                          }, 2200);
                        }
