/**
 * Indian Sign Language (ISL) Real-Time Interpreter
 * High-Speed 3D Kinematics & Gesture Translation Engine
 */

// Application State
const state = {
  cameraActive: false,
  isMirrored: true,
  showSkeleton: true,
  showFramingBox: false,
  enableTTS: true,
  activeMode: 'alphabet', // 'alphabet' | 'word' | 'sentence'
  stream: null,
  animationFrameId: null,
  isProcessing: false,
  lastRequestTime: 0,
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
  guideData: null,

  // Hands-Free Auto-Confirm Dwell State
  autoConfirmEnabled: true,
  holdTargetMs: 1200, // 1.2s steady sign hold to auto-commit
  activeHoldChar: null,
  holdStartTime: null,
  holdCooldownUntil: 0
};

// MediaPipe 21-hand landmark connections
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

// Smart Dictionary for Predictive Words & Autocomplete
const SMART_DICTIONARY = [
  // 2-Letter
  'AM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE', 'HI', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE',
  // 3-Letter
  'ACT', 'AGE', 'AIR', 'ALL', 'AND', 'ANY', 'ARM', 'ART', 'ASK', 'BAD', 'BAG', 'BAR', 'BAT', 'BED', 'BIG', 'BIT', 'BOX', 'BOY', 'BUS', 'BUY', 
  'CAB', 'CAN', 'CAP', 'CAR', 'CAT', 'COW', 'CRY', 'CUP', 'CUT', 
  'DAD', 'DAY', 'DOG', 'DRY', 'EAR', 'EAT', 'EGG', 'END', 'EYE', 'FAN', 'FAR', 'FAT', 'FEW', 'FIT', 'FLY', 'FOR', 'FOX', 'FUN', 
  'GAS', 'GET', 'GOD', 'GUN', 'GUY', 'HAT', 'HEN', 'HIT', 'HOT', 'HOW', 'HUG', 'ICE', 'ILL', 'INK', 'JAM', 'JAR', 'JAW', 'JOB', 'JOY', 
  'KEY', 'KID', 'LAW', 'LEG', 'LET', 'LIP', 'LOG', 'LOW', 'MAN', 'MAP', 'MOM', 'MUD', 'NET', 'NEW', 'NOT', 'NOW', 'NUT', 
  'OFF', 'OIL', 'OLD', 'ONE', 'OUR', 'OUT', 'PAN', 'PAY', 'PEN', 'PET', 'PIG', 'PIN', 'POT', 'PUT', 
  'RAN', 'RAT', 'RAW', 'RED', 'ROW', 'RUN', 'SAD', 'SAW', 'SAY', 'SEA', 'SEE', 'SET', 'SHE', 'SIT', 'SIX', 'SKY', 'SON', 'SUN', 
  'TEA', 'TEN', 'THE', 'TIE', 'TIN', 'TIP', 'TOE', 'TOP', 'TOY', 'TRY', 'TWO', 'USE', 'VAN', 'WAR', 'WAY', 'WET', 'WHO', 'WHY', 'WIN', 'YES', 'YET', 'YOU', 'ZOO',
  // 4-Letter
  'ABLE', 'BABY', 'BACK', 'BALL', 'BAND', 'BANK', 'BASE', 'BEAR', 'BEAT', 'BEST', 'BIRD', 'BLUE', 'BOAT', 'BODY', 'BONE', 'BOOK', 'BORN', 'BUSY',
  'CALL', 'CALM', 'CAME', 'CAMP', 'CARD', 'CARE', 'CART', 'CASE', 'CASH', 'CAST', 'CAVE', 'CHAT', 'CITY', 'COLD', 'COME', 'COOK', 'COOL',
  'DARK', 'DATA', 'DATE', 'DEAF', 'DEAR', 'DESK', 'DOOR', 'DRAW', 'DROP', 'DUTY', 'EASY', 'EDGE', 'EVEN', 'EVER', 'FACE', 'FACT', 'FAIR', 'FALL', 'FAST', 'FEEL', 'FILL', 'FILM', 'FIND', 'FINE', 'FIRE', 'FISH', 'FIVE', 'FOOD', 'FOOT', 'FREE', 'FROM', 'FULL', 'GAME', 'GIFT', 'GIRL', 'GIVE', 'GLAD', 'GOAL', 'GOLD', 'GOOD', 'GROW',
  'HAIR', 'HALF', 'HAND', 'HARD', 'HAVE', 'HEAD', 'HEAR', 'HEAT', 'HELP', 'HERE', 'HIGH', 'HOLD', 'HOME', 'HOPE', 'HOUR', 'HUGE', 'ICON', 'IDEA', 'INTO', 'ITEM', 'JOIN', 'JUMP', 'JUST', 'KEEP', 'KILL', 'KIND', 'KING', 'KNOW',
  'LADY', 'LAND', 'LAST', 'LATE', 'LEAD', 'LEFT', 'LESS', 'LIFE', 'LIFT', 'LIKE', 'LINE', 'LINK', 'LIST', 'LIVE', 'LOCK', 'LONG', 'LOOK', 'LORD', 'LOSE', 'LOSS', 'LOST', 'LOVE',
  'MADE', 'MAIL', 'MAIN', 'MAKE', 'MANY', 'MARK', 'MEAL', 'MEAN', 'MEET', 'MIND', 'MINE', 'MISS', 'MOON', 'MORE', 'MOST', 'MOVE', 'MUCH', 'MUST', 'NAME', 'NEAR', 'NECK', 'NEED', 'NEWS', 'NEXT', 'NICE', 'NINE', 'NOSE', 'NOTE',
  'ONCE', 'OPEN', 'OVER', 'PAGE', 'PAIN', 'PAIR', 'PARK', 'PART', 'PASS', 'PAST', 'PATH', 'PEAK', 'PLAN', 'PLAY', 'POEM', 'POET', 'POST', 'PULL', 'PURE', 'PUSH',
  'RACE', 'RAIN', 'READ', 'REAL', 'REST', 'RICE', 'RICH', 'RIDE', 'RING', 'RISE', 'RISK', 'ROAD', 'ROCK', 'ROLE', 'ROOF', 'ROOM', 'ROOT', 'ROSE', 'RULE',
  'SAFE', 'SAID', 'SAME', 'SAVE', 'SEAT', 'SEED', 'SEEK', 'SEEM', 'SEEN', 'SEND', 'SHIP', 'SHOE', 'SHOP', 'SHOT', 'SHOW', 'SHUT', 'SICK', 'SIDE', 'SIGN', 'SING', 'SITE', 'SIZE', 'SKIN', 'SLOW', 'SNOW', 'SOME', 'SONG', 'SOON', 'SOUL', 'SPOT', 'STAR', 'STAY', 'STEP', 'STOP', 'SUCH', 'SURE',
  'TAKE', 'TALK', 'TALL', 'TASK', 'TEAM', 'TELL', 'TERM', 'TEST', 'TEXT', 'THAN', 'THAT', 'THEM', 'THEN', 'THEY', 'THIN', 'THIS', 'TIME', 'TINY', 'TOLD', 'TOOK', 'TOOL', 'TOUR', 'TOWN', 'TREE', 'TRUE', 'TURN', 'TYPE',
  'UNIT', 'UPON', 'USER', 'VARY', 'VERY', 'VIEW', 'VOTE', 'WAIT', 'WAKE', 'WALK', 'WALL', 'WANT', 'WARM', 'WASH', 'WAVE', 'WEAR', 'WEEK', 'WELL', 'WENT', 'WEST', 'WHAT', 'WHEN', 'WIDE', 'WIFE', 'WILD', 'WILL', 'WIND', 'WISH', 'WITH', 'WOOD', 'WORD', 'WORK', 'YEAR', 'ZERO',
  // 5-Letter
  'ABOUT', 'ABOVE', 'ACROSS', 'ACTOR', 'ADULT', 'AFTER', 'AGAIN', 'AGENT', 'AGREE', 'AHEAD', 'ALONE', 'ALONG', 'ANGEL', 'ANGER', 'ANGRY', 'APPLE', 'APPLY', 'ARISE', 'AWAKE',
  'BASIC', 'BEACH', 'BEGAN', 'BEGIN', 'BEING', 'BELOW', 'BIRTH', 'BLACK', 'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BRAIN', 'BREAD', 'BREAK', 'BRING', 'BROWN', 'BUILD',
  'CARRY', 'CAUSE', 'CHAIN', 'CHAIR', 'CHEAP', 'CHECK', 'CHEST', 'CHIEF', 'CHILD', 'CHINA', 'CLAIM', 'CLASS', 'CLEAN', 'CLEAR', 'CLIMB', 'CLOCK', 'CLOSE', 'CLOUD', 'COACH', 'COAST', 'COLOR', 'COUNT', 'COURT', 'COVER', 'CRAFT', 'CROWD', 'CYCLE',
  'DAILY', 'DANCE', 'DEATH', 'DOUBT', 'DRAFT', 'DREAM', 'DRESS', 'DRINK', 'DRIVE', 'EARLY', 'EARTH', 'EMPTY', 'ENEMY', 'ENJOY', 'ENTER', 'EQUAL', 'ERROR', 'EVENT', 'EVERY',
  'FAITH', 'FALSE', 'FAULT', 'FIELD', 'FIFTH', 'FIGHT', 'FINAL', 'FIRST', 'FLOOR', 'FOCUS', 'FORCE', 'FORTH', 'FOUND', 'FRAME', 'FRESH', 'FRONT', 'FRUIT',
  'GIANT', 'GLASS', 'GLOBE', 'GRACE', 'GRAND', 'GRANT', 'GRASS', 'GREAT', 'GREEN', 'GROUP', 'GUARD', 'GUESS', 'GUEST', 'GUIDE',
  'HAPPY', 'HEART', 'HEAVY', 'HELLO', 'HORSE', 'HOTEL', 'HOUSE', 'HUMAN', 'IMAGE', 'INDEX', 'INNER', 'INPUT', 'ISSUE',
  'JUDGE', 'KNIFE', 'KNOWN', 'LABEL', 'LARGE', 'LAUGH', 'LEARN', 'LEAVE', 'LEGAL', 'LEVEL', 'LIGHT', 'LIMIT', 'LOCAL', 'LOGIC', 'LUNCH',
  'MAGIC', 'MAJOR', 'MAKER', 'MARCH', 'MATCH', 'MAYBE', 'MEDIA', 'METAL', 'MIGHT', 'MINOR', 'MODEL', 'MONEY', 'MONTH', 'MORAL', 'MOTOR', 'MOUNT', 'MOUSE', 'MOUTH', 'MOVIE', 'MUSIC',
  'NAKED', 'NEEDS', 'NEVER', 'NIGHT', 'NOISE', 'NORTH', 'NOTED', 'NOVEL', 'NURSE', 'OCCUR', 'OCEAN', 'OFFER', 'ORDER', 'OTHER', 'OUGHT',
  'PAINT', 'PANEL', 'PAPER', 'PARTY', 'PEACE', 'PHASE', 'PHONE', 'PHOTO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN', 'PLANE', 'PLANT', 'PLATE', 'POINT', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE', 'PRIME', 'PRIZE', 'PROOF', 'PROUD', 'PROVE',
  'QUEEN', 'QUICK', 'QUIET', 'QUITE', 'RADIO', 'RAISE', 'RANGE', 'RAPID', 'RATIO', 'REACH', 'READY', 'REPLY', 'RIGHT', 'RIVER', 'ROUGH', 'ROUND', 'ROUTE',
  'SCALE', 'SCENE', 'SCOPE', 'SCORE', 'SENSE', 'SERVE', 'SEVEN', 'SHALL', 'SHAPE', 'SHARE', 'SHARP', 'SHEET', 'SHIFT', 'SHINE', 'SHIRT', 'SHOCK', 'SHOOT', 'SHORT', 'SHOWN', 'SIGHT', 'SKILL', 'SLEEP', 'SLIDE', 'SMALL', 'SMART', 'SMILE', 'SMOKE', 'SOLID', 'SOLVE', 'SORRY', 'SOUND', 'SOUTH', 'SPACE', 'SPARE', 'SPEAK', 'SPEED', 'SPEND', 'SPLIT', 'SPORT', 'STAFF', 'STAGE', 'STAND', 'START', 'STATE', 'STEAM', 'STEEL', 'STICK', 'STILL', 'STOCK', 'STONE', 'STOOD', 'STORE', 'STORM', 'STORY', 'STUDY', 'STYLE', 'SUGAR', 'SWEET',
  'TABLE', 'TAKEN', 'TASTE', 'TEACH', 'TEETH', 'THANK', 'THEIR', 'THEME', 'THERE', 'THESE', 'THICK', 'THING', 'THINK', 'THIRD', 'THOSE', 'THREE', 'THROW', 'TIGHT', 'TIMES', 'TIRED', 'TITLE', 'TODAY', 'TOPIC', 'TOTAL', 'TOUCH', 'TOWER', 'TRACK', 'TRADE', 'TRAIN', 'TREAT', 'TREND', 'TRIAL', 'TRIED', 'TRUCK', 'TRULY', 'TRUST', 'TRUTH', 'TWICE',
  'UNDER', 'UNION', 'UNITY', 'UNTIL', 'UPPER', 'UPSET', 'URBAN', 'USAGE', 'USUAL', 'VALID', 'VALUE', 'VIDEO', 'VIRUS', 'VISIT', 'VITAL', 'VOICE',
  'WASTE', 'WATCH', 'WATER', 'WHEEL', 'WHERE', 'WHICH', 'WHILE', 'WHITE', 'WHOLE', 'WHOSE', 'WOMAN', 'WOMEN', 'WORLD', 'WORRY', 'WORSE', 'WORST', 'WORTH', 'WOULD', 'WOUND', 'WRITE', 'WRONG', 'YOUTH',
  // 6+ Letter Common ISL Words
  'ACTION', 'ALWAYS', 'BEAUTIFUL', 'BEFORE', 'BETTER', 'BROTHER', 'CHANGE', 'DOCTOR', 'FAMILY', 'FATHER', 'FRIEND', 'FUTURE', 'HEALTH', 'HOSPITAL', 'LISTEN', 'MORNING', 'MOTHER', 'PEOPLE', 'PLEASE', 'SCHOOL', 'SISTER', 'STUDENT', 'SUMMER', 'TEACHER', 'TOGETHER', 'WELCOME', 'WINDOW', 'WINTER', 'YESTERDAY'
];

// Contextual Predictive Word Chains for Sentence Builder
const SENTENCE_NEXT_WORDS = {
  '': ['I', 'YOU', 'HELLO', 'PLEASE', 'WHAT', 'HOW', 'THANK', 'GOOD', 'MY', 'WE', 'THE'],
  'I': ['AM', 'WANT', 'LIKE', 'NEED', 'CAN', 'HAVE', 'GO', 'KNOW', 'FEEL', 'SEE'],
  'I AM': ['HAPPY', 'FINE', 'LEARNING', 'READY', 'STUDENT', 'TIRED', 'GOOD', 'HERE'],
  'YOU': ['ARE', 'CAN', 'WANT', 'HAVE', 'KNOW', 'LOOK', 'HELP'],
  'YOU ARE': ['WELCOME', 'GOOD', 'KIND', 'BEAUTIFUL', 'READY', 'GREAT'],
  'HOW': ['ARE', 'IS', 'DO', 'CAN', 'MUCH'],
  'HOW ARE': ['YOU', 'THINGS', 'WE', 'THEY'],
  'THANK': ['YOU', 'EVERYONE', 'GOD'],
  'GOOD': ['MORNING', 'NIGHT', 'AFTERNOON', 'JOB', 'DAY', 'LUCK'],
  'WHAT': ['IS', 'ARE', 'DO', 'CAN', 'HAPPENED'],
  'WHAT IS': ['YOUR', 'THIS', 'NAME', 'THAT'],
  'MY': ['NAME', 'FRIEND', 'FAMILY', 'HOME', 'HAND', 'FATHER', 'MOTHER'],
  'MY NAME': ['IS'],
  'PLEASE': ['HELP', 'COME', 'WAIT', 'SIT', 'LISTEN', 'TELL']
};

// DOM Elements
const videoEl = document.getElementById('videoElement');
const canvasEl = document.getElementById('overlayCanvas');
const ctx = canvasEl.getContext('2d');
const cropCanvas = document.getElementById('cropCanvas');
const cropCtx = cropCanvas ? cropCanvas.getContext('2d') : null;

const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const framingBox = document.getElementById('framingBox');
const btnToggleCamera = document.getElementById('btnToggleCamera');
const btnPlaceholderStartCam = document.getElementById('btnPlaceholderStartCam');
const camBtnSvg = document.getElementById('camBtnSvg');
const camBtnText = document.getElementById('camBtnText');
const btnToggleSkeleton = document.getElementById('btnToggleSkeleton');
const skeletonBtnText = document.getElementById('skeletonBtnText');
const btnToggleBox = document.getElementById('btnToggleBox');
const boxBtnText = document.getElementById('boxBtnText');
const btnFlipCam = document.getElementById('btnFlipCam');

const detectionBadge = document.getElementById('detectionBadge');
const detectionText = document.getElementById('detectionText');
const fpsBadge = document.getElementById('fpsBadge');
const latencyBadge = document.getElementById('latencyBadge');
const isolationState = document.getElementById('isolationState');
const handsCountVal = document.getElementById('handsCountVal');
const trackingStatus = document.getElementById('trackingStatus');

const heroLetter = document.getElementById('heroLetter');
const letterName = document.getElementById('letterName');
const confBadge = document.getElementById('confBadge');
const confidenceBar = document.getElementById('confidenceBar');
const top3Row = document.getElementById('top3Row');

const btnToggleTTS = document.getElementById('btnToggleTTS');
const ttsSvgIcon = document.getElementById('ttsSvgIcon');
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

// Hands-Free Auto-Confirm & Suggestions DOM (Word Mode)
const wordHoldConfirmWrapper = document.getElementById('wordHoldConfirmWrapper');
const wordHoldConfirmText = document.getElementById('wordHoldConfirmText');
const wordHoldConfirmTimer = document.getElementById('wordHoldConfirmTimer');
const wordHoldProgressBar = document.getElementById('wordHoldProgressBar');
const wordSuggestionsBox = document.getElementById('wordSuggestionsBox');
const wordSuggestionsRow = document.getElementById('wordSuggestionsRow');
const wordSuggestionsHint = document.getElementById('wordSuggestionsHint');

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

// Hands-Free Auto-Confirm & Suggestions DOM (Sentence Mode)
const sentenceHoldConfirmWrapper = document.getElementById('sentenceHoldConfirmWrapper');
const sentenceHoldConfirmText = document.getElementById('sentenceHoldConfirmText');
const sentenceHoldConfirmTimer = document.getElementById('sentenceHoldConfirmTimer');
const sentenceHoldProgressBar = document.getElementById('sentenceHoldProgressBar');
const sentenceSuggestionsBox = document.getElementById('sentenceSuggestionsBox');
const sentenceSuggestionsRow = document.getElementById('sentenceSuggestionsRow');
const sentenceSuggestionsTitle = document.getElementById('sentenceSuggestionsTitle');

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
  initCropCanvasPlaceholder();
  if (framingBox) framingBox.style.display = 'none';
});

function initCropCanvasPlaceholder() {
  if (!cropCtx) return;
  cropCtx.fillStyle = '#080a0d';
  cropCtx.fillRect(0, 0, 120, 120);
  cropCtx.fillStyle = '#374151';
  cropCtx.font = '10px Inter, sans-serif';
  cropCtx.textAlign = 'center';
  cropCtx.fillText('No Hand', 60, 65);
}

function initEventListeners() {
  btnToggleCamera.addEventListener('click', toggleCamera);
  btnPlaceholderStartCam.addEventListener('click', toggleCamera);

  btnToggleSkeleton.addEventListener('click', () => {
    state.showSkeleton = !state.showSkeleton;
    skeletonBtnText.textContent = `Skeleton: ${state.showSkeleton ? 'ON' : 'OFF'}`;
  });

  btnToggleBox.addEventListener('click', () => {
    state.showFramingBox = !state.showFramingBox;
    if (framingBox) framingBox.style.display = state.showFramingBox ? 'block' : 'none';
    boxBtnText.textContent = `Guide Box: ${state.showFramingBox ? 'ON' : 'OFF'}`;
  });

  btnFlipCam.addEventListener('click', () => {
    state.isMirrored = !state.isMirrored;
    videoEl.style.transform = state.isMirrored ? 'scaleX(-1)' : 'scaleX(1)';
  });

  btnToggleTTS.addEventListener('click', () => {
    state.enableTTS = !state.enableTTS;
    ttsText.textContent = `Audio: ${state.enableTTS ? 'ON' : 'OFF'}`;
    ttsSvgIcon.style.opacity = state.enableTTS ? '1' : '0.4';
  });

  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      switchMode(tab.dataset.mode);
    });
  });

  // Alphabet Mode Actions
  btnSpeakAlpha.addEventListener('click', () => {
    if (state.currentSmoothedLetter && state.currentSmoothedLetter !== '-') {
      speakText(`Letter ${state.currentSmoothedLetter}`);
    }
  });

  btnCopyAlpha.addEventListener('click', () => {
    if (state.currentSmoothedLetter && state.currentSmoothedLetter !== '-') {
      navigator.clipboard.writeText(state.currentSmoothedLetter);
      showToast(`Copied "${state.currentSmoothedLetter}" to clipboard`);
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
    if (state.assembledWord) {
      navigator.clipboard.writeText(state.assembledWord);
      showToast(`Copied "${state.assembledWord}"`);
    }
  });

  btnAddWordToSentence.addEventListener('click', () => {
    if (!state.assembledWord) return;
    switchMode('sentence');
    if (!sentenceWorkspace.style.display || sentenceWorkspace.style.display === 'none') {
      initSentenceMode(3);
    }
    state.completedWords.push(state.assembledWord);
    updateSentenceOutput();
    showToast(`Word "${state.assembledWord}" added to sentence`);
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
    if (state.fullSentence) {
      navigator.clipboard.writeText(state.fullSentence);
      showToast('Sentence copied to clipboard');
    }
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
   CAMERA & ZERO-LAG STREAMING PIPELINE
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

    camBtnSvg.innerHTML = '<rect x="4" y="4" width="16" height="16" rx="2"/>';
    camBtnText.textContent = 'Stop Camera';
    btnToggleCamera.classList.add('btn-accent');
    btnToggleCamera.classList.remove('btn-primary');

    detectionBadge.className = 'badge badge-success';
    detectionText.textContent = 'Tracking Active';

    startHighSpeedLoop();
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
  if (state.processingInterval) {
    clearInterval(state.processingInterval);
    state.processingInterval = null;
  }

  state.cameraActive = false;
  cameraPlaceholder.style.display = 'flex';

  camBtnSvg.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  camBtnText.textContent = 'Start Camera';
  btnToggleCamera.classList.remove('btn-accent');
  btnToggleCamera.classList.add('btn-primary');

  detectionBadge.className = 'badge';
  detectionText.textContent = 'Camera Off';
  fpsBadge.textContent = '0 FPS';
  if (latencyBadge) latencyBadge.textContent = '0 ms';

  handleInferenceResult({
    hand_detected: false,
    letter: null,
    smoothed_letter: '-',
    confidence: 0.0,
    top3: [],
    landmarks: [],
    bbox: null,
    hand_count: 0
  });

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

/**
 * Reliable Full-Fidelity Vision Loop
 * Captures clean 640x480 frames so MediaPipe palm detection locks onto hands 100% reliably.
 */
function startHighSpeedLoop() {
  const offscreen = document.createElement('canvas');
  const offCtx = offscreen.getContext('2d');
  let isRequestPending = false;

  state.processingInterval = setInterval(async () => {
    if (!state.cameraActive || isRequestPending) return;

    // FPS Meter
    state.fpsCounter++;
    const now = performance.now();
    if (now - state.lastFpsTimestamp >= 1000) {
      state.currentFps = state.fpsCounter;
      fpsBadge.textContent = `${state.currentFps} FPS`;
      state.fpsCounter = 0;
      state.lastFpsTimestamp = now;
    }

    // High-performance standardized transmission dimension (480px width - instantaneous encoding & transfer)
    const targetW = 480;
    const vW = videoEl.videoWidth || 640;
    const vH = videoEl.videoHeight || 480;
    const targetH = Math.round(targetW * (vH / vW));

    if (offscreen.width !== targetW || offscreen.height !== targetH) {
      offscreen.width = targetW;
      offscreen.height = targetH;
    }
    offCtx.drawImage(videoEl, 0, 0, targetW, targetH);
    const base64Data = offscreen.toDataURL('image/jpeg', 0.58);

    isRequestPending = true;
    const reqStart = performance.now();
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data, use_fallback: false })
      });

      if (resp.ok) {
        const data = await resp.json();
        const roundTrip = Math.round(performance.now() - reqStart);
        if (latencyBadge) latencyBadge.textContent = `${roundTrip} ms`;
        handleInferenceResult(data);
      }
    } catch (err) {
      console.warn('Inference error:', err);
    } finally {
      isRequestPending = false;
    }
  }, 40);
}

function handleInferenceResult(data) {
  state.handDetected = data.hand_detected;
  state.currentLetter = data.letter || '-';
  state.currentSmoothedLetter = data.smoothed_letter || '-';
  state.currentConfidence = data.confidence || 0.0;
  state.top3 = data.top3 || [];
  state.handsCount = data.hand_count || 0;

  // Update Hand Crop Preview
  const cropImg = document.getElementById('cropPreviewImg');
  if (cropImg && data.hand_crop_base64) {
    cropImg.src = data.hand_crop_base64;
  }

  // Update Status Badges (No Emojis)
  handsCountVal.textContent = state.handsCount;

  if (state.handDetected) {
    isolationState.className = 'badge badge-success';
    isolationState.textContent = 'Body Filtered';
    trackingStatus.textContent = `${state.handsCount} Hand(s) Locked`;
    detectionBadge.className = 'badge badge-success';
    detectionText.textContent = `Active (${state.handsCount} Hand${state.handsCount > 1 ? 's' : ''})`;
  } else {
    isolationState.className = 'badge';
    isolationState.textContent = 'Standby';
    trackingStatus.textContent = 'Standby';
    detectionBadge.className = 'badge';
    detectionText.textContent = state.cameraActive ? 'Awaiting Hand' : 'Camera Off';
  }

  // Update Prediction Hero Box
  if (state.handDetected && state.currentSmoothedLetter !== '-') {
    heroLetter.textContent = state.currentSmoothedLetter;
    const pct = Math.round(state.currentConfidence * 100);
    confBadge.textContent = `${pct}% Conf`;
    confidenceBar.style.width = `${pct}%`;
    letterName.textContent = `Recognized: Letter ${state.currentSmoothedLetter}`;
  } else {
    heroLetter.textContent = '-';
    confBadge.textContent = '0% Conf';
    confidenceBar.style.width = '0%';
    letterName.textContent = state.cameraActive
      ? 'Awaiting Hand In Frame...'
      : 'Camera Inactive';
  }

  // Update Top-3 Pills
  top3Row.innerHTML = '';
  if (state.handDetected && state.top3.length > 0) {
    state.top3.forEach(item => {
      const pill = document.createElement('span');
      pill.className = 'top3-pill font-mono';
      pill.textContent = `${item.letter}: ${Math.round(item.prob * 100)}%`;
      top3Row.appendChild(pill);
    });
  } else {
    top3Row.innerHTML = '<span class="top3-pill font-mono">-</span><span class="top3-pill font-mono">-</span><span class="top3-pill font-mono">-</span>';
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

  // Draw bounding box
  if (bbox && bbox.w > 0 && state.handDetected) {
    ctx.save();
    let bx1 = bbox.x1;
    let bx2 = bbox.x2 || (bbox.x1 + bbox.w);
    let bw = bbox.w;
    let by1 = bbox.y1;
    let bh = bbox.h;

    if (state.isMirrored) {
      bx1 = w - bx2;
    }

    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bx1, by1, bw, bh);

    ctx.fillStyle = '#2563eb';
    ctx.font = '500 11px Inter, sans-serif';
    ctx.fillText('ISOLATED HAND', bx1 + 4, Math.max(14, by1 - 4));
    ctx.restore();
  }

  // Draw skeletal landmarks
  if (landmarksList && landmarksList.length > 0) {
    landmarksList.forEach((hand) => {
      // Draw Connections (Bones)
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.75)';
      ctx.lineWidth = 2.2;
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
        ctx.arc(lx, ly, 3.5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
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
  state.activeHoldChar = null;
  state.holdStartTime = null;
  modeTabs.forEach(t => {
    t.classList.toggle('active', t.dataset.mode === newMode);
  });
  paneAlphabet.classList.toggle('active', newMode === 'alphabet');
  paneWord.classList.toggle('active', newMode === 'word');
  paneSentence.classList.toggle('active', newMode === 'sentence');

  if (newMode === 'word') {
    updateWordSuggestions();
  } else if (newMode === 'sentence') {
    updateSentenceSuggestions();
  }
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
    bigAlphaDisplay.textContent = '-';
    bigAlphaDesc.textContent = state.cameraActive
      ? 'Perform an ISL sign with your hand in front of the camera.'
      : 'Start camera to begin interpretation.';
    return;
  }

  bigAlphaDisplay.textContent = letter;

  if (state.guideData && state.guideData[letter]) {
    bigAlphaDesc.textContent = state.guideData[letter];
  } else {
    bigAlphaDesc.textContent = `Letter "${letter}" (${Math.round(confidence * 100)}% confidence)`;
  }

  // Append to history if stable
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
    alphaHistoryContainer.innerHTML = '<span class="history-empty">Signs will log here as you perform them...</span>';
    return;
  }

  alphaHistoryContainer.innerHTML = '';
  state.alphaHistory.forEach(item => {
    const chip = document.createElement('span');
    chip.className = 'history-chip font-mono';
    chip.textContent = item.letter;
    alphaHistoryContainer.appendChild(chip);
  });
}

/* ==========================================================
   AUDIO SYNTHESIS FEEDBACK
   ========================================================== */
function playChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const audioCtx = new AudioCtx();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
    osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.10); // A5
    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.16);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.16);
  } catch (e) {
    // Audio context silenced or blocked by autoplay
  }
}

/* ==========================================================
   HANDS-FREE AUTO-CONFIRM DWELL ENGINE
   ========================================================== */
function processAutoHoldConfirm(mode, letter, confidence, handDetected) {
  if (!state.autoConfirmEnabled) return;

  const now = performance.now();
  if (now < state.holdCooldownUntil) {
    updateHoldProgress(mode, 0, 0, null, 'Cooldown...');
    return;
  }

  // Require solid hand detection and a clean sign
  if (!handDetected || !letter || letter === '-' || confidence < 0.35) {
    state.activeHoldChar = null;
    state.holdStartTime = null;
    updateHoldProgress(mode, 0, 0, null, 'Hold sign steadily to auto-confirm');
    return;
  }

  // If in Word Mode and word is already complete, do not advance
  if (mode === 'word' && state.activeWordSlotIndex >= state.wordTargetLength) {
    updateHoldProgress(mode, 0, 0, null, 'Word complete');
    return;
  }

  if (letter === state.activeHoldChar) {
    if (!state.holdStartTime) {
      state.holdStartTime = now;
    }
    const elapsed = now - state.holdStartTime;
    const pct = Math.min(100, Math.round((elapsed / state.holdTargetMs) * 100));
    updateHoldProgress(mode, pct, elapsed, letter, `Holding '${letter}'...`);

    if (elapsed >= state.holdTargetMs) {
      // Trigger Auto-Confirmation!
      playChime();
      triggerHoldFlash(mode);

      if (mode === 'word') {
        saveActiveWordSlot();
      } else if (mode === 'sentence') {
        saveSentenceLetter();
      }

      // Enter refractory cooldown (750ms) so user can comfortably switch hands
      state.holdCooldownUntil = performance.now() + 750;
      state.activeHoldChar = null;
      state.holdStartTime = null;
    }
  } else {
    // New character started
    state.activeHoldChar = letter;
    state.holdStartTime = now;
    updateHoldProgress(mode, 0, 0, letter, `Holding '${letter}'...`);
  }
}

function updateHoldProgress(mode, pct, elapsed, char, label) {
  const progressBar = mode === 'word' ? wordHoldProgressBar : sentenceHoldProgressBar;
  const timerEl = mode === 'word' ? wordHoldConfirmTimer : sentenceHoldConfirmTimer;
  const textEl = mode === 'word' ? wordHoldConfirmText : sentenceHoldConfirmText;

  if (progressBar) {
    progressBar.style.width = `${pct}%`;
  }
  if (timerEl) {
    const s = (elapsed / 1000).toFixed(1);
    const targetS = (state.holdTargetMs / 1000).toFixed(1);
    timerEl.textContent = `${s}s / ${targetS}s`;
  }
  if (textEl) {
    const icon = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
    if (char && pct > 0) {
      textEl.innerHTML = `${icon} Auto-Locking '<strong>${char}</strong>' (${pct}%)`;
    } else {
      textEl.innerHTML = `${icon} ${label}`;
    }
  }
}

function triggerHoldFlash(mode) {
  const progressBar = mode === 'word' ? wordHoldProgressBar : sentenceHoldProgressBar;
  if (progressBar) {
    progressBar.classList.add('confirm-flash');
    setTimeout(() => {
      progressBar.classList.remove('confirm-flash');
      progressBar.style.width = '0%';
    }, 250);
  }
}

/* ==========================================================
   MODE 2: WORD BUILDER & SMART PREDICTIONS
   ========================================================== */
function initWordMode(length) {
  state.wordTargetLength = Math.max(1, Math.min(15, length));
  state.activeWordSlotIndex = 0;
  state.wordSlots = [];
  state.assembledWord = '';
  state.activeHoldChar = null;
  state.holdStartTime = null;

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
  updateWordSuggestions();
}

function renderWordSlots() {
  wordSlotsContainer.innerHTML = '';
  state.wordSlots.forEach((slot, idx) => {
    const slotCard = document.createElement('div');
    slotCard.className = 'slot-card font-mono';
    if (idx === state.activeWordSlotIndex) {
      slotCard.classList.add('active');
    }
    if (slot.locked) {
      slotCard.classList.add('completed');
    }

    const numEl = document.createElement('div');
    numEl.className = 'slot-number font-mono';
    numEl.textContent = `#${slot.index}`;

    const charEl = document.createElement('div');
    charEl.className = 'slot-char font-mono';
    charEl.textContent = slot.char || (idx === state.activeWordSlotIndex ? '-' : '_');

    slotCard.appendChild(numEl);
    slotCard.appendChild(charEl);

    slotCard.addEventListener('click', () => {
      state.activeWordSlotIndex = idx;
      renderWordSlots();
      updateWordActiveSlotHUD();
      updateWordSuggestions();
    });

    wordSlotsContainer.appendChild(slotCard);
  });
}

function updateWordActiveSlotHUD() {
  const currentSlotNum = state.activeWordSlotIndex + 1;
  activeSlotIndicator.textContent = `Slot ${currentSlotNum} of ${state.wordTargetLength}`;

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
          if (charEl) charEl.textContent = '-';
        }
      }
    }
  }

  // Process hands-free dwell hold
  processAutoHoldConfirm('word', letter, confidence, handDetected);
}

function saveActiveWordSlot() {
  if (state.activeWordSlotIndex >= state.wordTargetLength) return;

  if (!state.handDetected || !state.currentSmoothedLetter || state.currentSmoothedLetter === '-') {
    showToast('Perform a clear hand sign before saving');
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
    updateWordSuggestions();
  }
}

function prevActiveWordSlot() {
  if (state.activeWordSlotIndex > 0) {
    state.activeWordSlotIndex--;
    state.wordSlots[state.activeWordSlotIndex].locked = false;
    renderWordSlots();
    updateWordActiveSlotHUD();
    updateWordSuggestions();
  }
}

function finalizeWord() {
  renderWordSlots();
  state.assembledWord = state.wordSlots.map(s => s.char).join('');
  assembledWordDisplay.textContent = state.assembledWord;
  wordResultCard.style.display = 'flex';
  activeSlotIndicator.textContent = `Word Complete (${state.wordTargetLength} Letters)`;

  if (wordSuggestionsRow) {
    wordSuggestionsRow.innerHTML = `<span class="suggestions-empty">Completed word: <strong>${state.assembledWord}</strong></span>`;
  }

  if (state.enableTTS) {
    speakText(`Word: ${state.assembledWord}`);
  }
}

/**
 * Smart Word Suggestions Engine
 * e.g., If letters 'C' and 'A' are locked in 3-letter mode -> suggests CAT, CAB, CAP, CAN, CAR
 */
function updateWordSuggestions() {
  if (!wordSuggestionsRow) return;

  const lockedChars = state.wordSlots.slice(0, state.activeWordSlotIndex).map(s => s.char).join('').toUpperCase();
  const targetLen = state.wordTargetLength;

  let matches = [];

  if (lockedChars.length === 0) {
    // Show popular starter words matching target length
    matches = SMART_DICTIONARY.filter(w => w.length === targetLen).slice(0, 8);
    if (wordSuggestionsHint) {
      wordSuggestionsHint.textContent = `Popular ${targetLen}-letter words (tap to autofill)`;
    }
  } else {
    if (wordSuggestionsHint) {
      wordSuggestionsHint.textContent = `Suggestions for prefix "${lockedChars}" (tap to autofill)`;
    }

    // 1. Exact target length matches (highest priority)
    const exactLen = SMART_DICTIONARY.filter(w => w.startsWith(lockedChars) && w.length === targetLen);
    // 2. Other length matches starting with prefix
    const otherLen = SMART_DICTIONARY.filter(w => w.startsWith(lockedChars) && w.length !== targetLen);

    matches = [...exactLen, ...otherLen].slice(0, 10);
  }

  wordSuggestionsRow.innerHTML = '';
  if (matches.length === 0) {
    wordSuggestionsRow.innerHTML = '<span class="suggestions-empty">No word predictions found for this letter combination</span>';
    return;
  }

  matches.forEach(word => {
    const chip = document.createElement('button');
    chip.className = 'suggestion-chip font-mono';
    chip.type = 'button';
    if (lockedChars.length > 0 && word.startsWith(lockedChars)) {
      chip.innerHTML = `<span class="chip-key">${lockedChars}</span>${word.slice(lockedChars.length)}`;
    } else {
      chip.textContent = word;
    }
    chip.title = `Click to autofill "${word}"`;
    chip.addEventListener('click', () => {
      autofillWord(word);
    });
    wordSuggestionsRow.appendChild(chip);
  });
}

function autofillWord(word) {
  const letters = word.toUpperCase().split('');
  state.wordTargetLength = letters.length;
  if (numLettersInput) numLettersInput.value = letters.length;
  state.wordSlots = [];

  letters.forEach((char, i) => {
    state.wordSlots.push({
      index: i + 1,
      char: char,
      locked: true
    });
  });

  state.activeWordSlotIndex = letters.length;
  finalizeWord();
  playChime();
  showToast(`Auto-filled "${word}"!`);
}

/* ==========================================================
   MODE 3: SENTENCE MODE & CONTEXTUAL PREDICTIONS
   ========================================================== */
function initSentenceMode(targetWords) {
  state.sentenceTargetWords = Math.max(1, Math.min(10, targetWords));
  state.currentWordIndex = 0;
  state.completedWords = [];
  state.currentWordLetters = [];
  state.fullSentence = '';
  state.activeHoldChar = null;
  state.holdStartTime = null;

  sentenceConfigBox.style.display = 'block';
  sentenceWorkspace.style.display = 'flex';

  renderSentenceBreadcrumbs();
  renderSentenceWordSlots();
  updateSentenceOutput();
  updateSentenceSuggestions();
}

function renderSentenceBreadcrumbs() {
  wordsBreadcrumbBar.innerHTML = '';
  for (let i = 0; i < state.sentenceTargetWords; i++) {
    const pill = document.createElement('div');
    pill.className = 'word-breadcrumb-pill font-mono';
    if (i === state.currentWordIndex) {
      pill.classList.add('active');
      pill.textContent = `Word ${i + 1}`;
    } else if (i < state.currentWordIndex) {
      pill.classList.add('completed');
      pill.textContent = state.completedWords[i] || `Word ${i + 1}`;
    } else {
      pill.textContent = `Word ${i + 1}`;
    }
    wordsBreadcrumbBar.appendChild(pill);
  }
}

function renderSentenceWordSlots() {
  activeWordTitle.textContent = `Word ${state.currentWordIndex + 1} of ${state.sentenceTargetWords}`;
  wordLetterCountBadge.textContent = `${state.currentWordLetters.length} Letters`;

  sentenceLetterSlots.innerHTML = '';
  if (state.currentWordLetters.length === 0) {
    const emptySlot = document.createElement('div');
    emptySlot.className = 'sentence-char-pill font-mono';
    emptySlot.style.borderColor = 'var(--border-active)';
    emptySlot.textContent = '-';
    sentenceLetterSlots.appendChild(emptySlot);
  } else {
    state.currentWordLetters.forEach((char) => {
      const slot = document.createElement('div');
      slot.className = 'sentence-char-pill font-mono';
      slot.textContent = char;
      sentenceLetterSlots.appendChild(slot);
    });
    const activeSlot = document.createElement('div');
    activeSlot.className = 'sentence-char-pill font-mono';
    activeSlot.style.borderColor = 'var(--border-active)';
    activeSlot.textContent = '-';
    sentenceLetterSlots.appendChild(activeSlot);
  }
}

function updateSentenceModePreview(letter, confidence, handDetected) {
  if (handDetected && letter !== '-') {
    sentenceInterpretingVal.textContent = letter;
    const activePill = sentenceLetterSlots.lastElementChild;
    if (activePill) activePill.textContent = letter;
  } else {
    sentenceInterpretingVal.textContent = '-';
    const activePill = sentenceLetterSlots.lastElementChild;
    if (activePill) activePill.textContent = '-';
  }

  // Process hands-free dwell hold
  processAutoHoldConfirm('sentence', letter, confidence, handDetected);
}

function saveSentenceLetter() {
  if (!state.handDetected || !state.currentSmoothedLetter || state.currentSmoothedLetter === '-') {
    showToast('Perform a clear hand sign before saving');
    return;
  }

  const letterToSave = state.currentSmoothedLetter;
  state.currentWordLetters.push(letterToSave);
  if (state.enableTTS) {
    speakText(letterToSave);
  }
  renderSentenceWordSlots();
  updateSentenceSuggestions();
}

function backSentenceLetter() {
  if (state.currentWordLetters.length > 0) {
    state.currentWordLetters.pop();
    renderSentenceWordSlots();
    updateSentenceSuggestions();
  }
}

function completeCurrentWordAndAdvance() {
  if (state.currentWordLetters.length === 0) {
    showToast('Add at least one letter before completing the word');
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
    activeWordTitle.textContent = 'Sentence Complete';
    sentenceLetterSlots.innerHTML = '';
    if (sentenceSuggestionsRow) {
      sentenceSuggestionsRow.innerHTML = '<span class="suggestions-empty">Sentence completed successfully!</span>';
    }
    if (state.enableTTS) {
      setTimeout(() => speakText(`Complete sentence: ${state.fullSentence}`), 600);
    }
  } else {
    renderSentenceBreadcrumbs();
    renderSentenceWordSlots();
    updateSentenceSuggestions();
  }
}

/**
 * Smart Sentence Predictions Engine
 * Handles both prefix completions (e.g. HE -> HELLO, HELP) and contextual next-words
 */
function updateSentenceSuggestions() {
  if (!sentenceSuggestionsRow) return;

  const currentPrefix = state.currentWordLetters.join('').toUpperCase();

  if (currentPrefix.length > 0) {
    // Current word prefix completion
    if (sentenceSuggestionsTitle) {
      sentenceSuggestionsTitle.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Complete Word "${currentPrefix}"`;
    }

    const matches = SMART_DICTIONARY.filter(w => w.startsWith(currentPrefix)).slice(0, 10);
    renderSentenceChips(matches, (word) => {
      state.completedWords.push(word);
      state.currentWordLetters = [];
      state.currentWordIndex++;
      if (state.enableTTS) speakText(`Word: ${word}`);
      playChime();
      updateSentenceOutput();
      checkSentenceComplete();
    });
  } else {
    // Next-Word prediction based on previous words
    if (sentenceSuggestionsTitle) {
      sentenceSuggestionsTitle.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Word Predictions`;
    }

    let nextWords = [];
    if (state.completedWords.length === 0) {
      nextWords = SENTENCE_NEXT_WORDS[''] || ['I', 'YOU', 'HELLO', 'PLEASE', 'WHAT', 'HOW', 'THANK', 'GOOD'];
    } else {
      const len = state.completedWords.length;
      const lastTwo = len >= 2 ? `${state.completedWords[len-2]} ${state.completedWords[len-1]}` : null;
      const lastOne = state.completedWords[len-1];

      if (lastTwo && SENTENCE_NEXT_WORDS[lastTwo]) {
        nextWords = SENTENCE_NEXT_WORDS[lastTwo];
      } else if (SENTENCE_NEXT_WORDS[lastOne]) {
        nextWords = SENTENCE_NEXT_WORDS[lastOne];
      } else {
        nextWords = ['IS', 'ARE', 'AND', 'THE', 'TO', 'YOU', 'ME', 'GOOD', 'HERE'];
      }
    }

    renderSentenceChips(nextWords.slice(0, 10), (word) => {
      state.completedWords.push(word);
      state.currentWordLetters = [];
      state.currentWordIndex++;
      if (state.enableTTS) speakText(`Word: ${word}`);
      playChime();
      updateSentenceOutput();
      checkSentenceComplete();
    });
  }
}

function renderSentenceChips(wordList, onSelect) {
  sentenceSuggestionsRow.innerHTML = '';
  if (!wordList || wordList.length === 0) {
    sentenceSuggestionsRow.innerHTML = '<span class="suggestions-empty">Sign a letter to view completions</span>';
    return;
  }

  wordList.forEach(word => {
    const chip = document.createElement('button');
    chip.className = 'suggestion-chip font-mono';
    chip.type = 'button';
    chip.textContent = word;
    chip.title = `Click to add "${word}" to sentence`;
    chip.addEventListener('click', () => {
      onSelect(word);
    });
    sentenceSuggestionsRow.appendChild(chip);
  });
}

function checkSentenceComplete() {
  if (state.currentWordIndex >= state.sentenceTargetWords) {
    renderSentenceBreadcrumbs();
    activeWordTitle.textContent = 'Sentence Complete';
    sentenceLetterSlots.innerHTML = '';
    if (sentenceSuggestionsRow) {
      sentenceSuggestionsRow.innerHTML = '<span class="suggestions-empty">Sentence completed successfully!</span>';
    }
    if (state.enableTTS) {
      setTimeout(() => speakText(`Complete sentence: ${state.fullSentence}`), 600);
    }
  } else {
    renderSentenceBreadcrumbs();
    renderSentenceWordSlots();
    updateSentenceSuggestions();
  }
}

function updateSentenceOutput() {
  state.fullSentence = state.completedWords.join(' ');
  if (state.fullSentence) {
    fullSentenceDisplay.textContent = state.fullSentence;
    sentenceCharCount.textContent = `${state.fullSentence.length} chars · ${state.completedWords.length} words`;
  } else {
    fullSentenceDisplay.textContent = 'Sentence output will assemble here...';
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
    utterance.rate = 1.05;
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
        modelStatusText.textContent = `Model Ready (${info.num_classes} Classes · ${info.accuracy}% Acc)`;
      }
    }
  } catch (err) {
    // Suppress
  }

  try {
    const res2 = await fetch('/api/reference');
    if (res2.ok) {
      state.guideData = await res2.json();
      renderGuideModal();
    }
  } catch (err) {
    // Suppress
  }
}

function openGuideModal() {
  guideModal.style.display = 'flex';
}

function closeGuideModal() {
  guideModal.style.display = 'none';
}

function renderGuideModal() {
  if (!state.guideData) return;
  guideGrid.innerHTML = '';

  Object.keys(state.guideData).sort().forEach(letter => {
    const desc = state.guideData[letter];
    const card = document.createElement('div');
    card.className = 'guide-card';
    card.innerHTML = `
      <div class="guide-card-letter font-mono">${letter}</div>
      <div class="guide-card-desc">${desc}</div>
    `;
    guideGrid.appendChild(card);
  });
}

function showToast(msg) {
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.bottom = '20px';
  toast.style.right = '20px';
  toast.style.background = '#181d26';
  toast.style.border = '1px solid #2f3645';
  toast.style.color = '#f3f4f6';
  toast.style.padding = '9px 16px';
  toast.style.borderRadius = '6px';
  toast.style.fontWeight = '500';
  toast.style.fontSize = '0.82rem';
  toast.style.boxShadow = '0 8px 24px rgba(0,0,0,0.5)';
  toast.style.zIndex = '9999';
  toast.style.transition = 'opacity 0.25s ease';
  toast.textContent = msg;

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 250);
  }, 1800);
}
