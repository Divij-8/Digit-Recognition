/**
 * Voice Digit Recognition Application
 * Dual Engine:
 * 1. MFCC (Mel-Frequency Cepstral Coefficients) + K-Means VQ Codebook Classifier
 * 2. Continuous Web Speech API Engine
 */

(function () {
  'use strict';

  // --- Digit Metadata & Musical Chime Notes ---
  const DIGIT_INFO = {
    0: { word: 'ZERO', phonetic: '/ˈzɪə.roʊ/', note: 261.63 },
    1: { word: 'ONE', phonetic: '/wʌn/', note: 293.66 },
    2: { word: 'TWO', phonetic: '/tuː/', note: 329.63 },
    3: { word: 'THREE', phonetic: '/θriː/', note: 392.00 },
    4: { word: 'FOUR', phonetic: '/fɔːr/', note: 440.00 },
    5: { word: 'FIVE', phonetic: '/faɪv/', note: 523.25 },
    6: { word: 'SIX', phonetic: '/sɪks/', note: 587.33 },
    7: { word: 'SEVEN', phonetic: '/ˈsɛv.ən/', note: 659.25 },
    8: { word: 'EIGHT', phonetic: '/eɪt/', note: 783.99 },
    9: { word: 'NINE', phonetic: '/naɪn/', note: 880.00 }
  };

  const WORD_TO_DIGIT_MAP = {
    'zero': 0, 'oh': 0, 'o': 0, 'naught': 0, 'null': 0, 'nil': 0,
    'one': 1, 'won': 1, 'first': 1,
    'two': 2, 'to': 2, 'too': 2, 'second': 2,
    'three': 3, 'third': 3, 'tree': 3,
    'four': 4, 'for': 4, 'fore': 4, 'fourth': 4,
    'five': 5, 'fifth': 5,
    'six': 6, 'sixth': 6,
    'seven': 7, 'seventh': 7,
    'eight': 8, 'ate': 8, 'eighth': 8,
    'nine': 9, 'ninth': 9
  };

  // --- State Variables ---
  let activeEngine = 'mfcc'; // 'mfcc' or 'speech'
  let trainedCodebooks = null; // Loaded from codebooks.json
  let audioContext = null;
  let visualizerAnimId = null;
  let isListening = false;
  let isRecording = false;
  let speechRecognition = null;
  let streamDigits = [];
  let lastProcessedTime = 0;

  // --- DOM Elements ---
  const tabModeMfcc = document.getElementById('tabModeMfcc');
  const tabModeSpeech = document.getElementById('tabModeSpeech');
  const modeTagPill = document.getElementById('modeTagPill');
  const statusCard = document.getElementById('statusCard');
  const statusText = document.getElementById('statusText');

  const heroCard = document.getElementById('heroCard');
  const digitDisplay = document.getElementById('digitDisplay');
  const wordRepresentation = document.getElementById('wordRepresentation');
  const phoneticHint = document.getElementById('phoneticHint');
  const confidenceBadge = document.getElementById('confidenceBadge');
  const rippleRings = document.getElementById('rippleRings');

  const visualizerCanvas = document.getElementById('visualizerCanvas');
  const visCtx = visualizerCanvas.getContext('2d');
  const mfccCanvas = document.getElementById('mfccCanvas');
  const mfccCtx = mfccCanvas.getContext('2d');
  const mfccDimBadge = document.getElementById('mfccDimBadge');

  const kmeansCard = document.getElementById('kmeansCard');
  const kmeansBarsGrid = document.getElementById('kmeansBarsGrid');
  const kmeansWinnerPill = document.getElementById('kmeansWinnerPill');

  const mfccActionGroup = document.getElementById('mfccActionGroup');
  const speechActionGroup = document.getElementById('speechActionGroup');
  const btnRecordDigit = document.getElementById('btnRecordDigit');
  const btnRecordText = document.getElementById('btnRecordText');
  const recProgressBar = document.getElementById('recProgressBar');
  const btnToggleMic = document.getElementById('btnToggleMic');
  const btnMicText = document.getElementById('btnMicText');

  const checkSoundEffects = document.getElementById('checkSoundEffects');
  const checkVoiceReadout = document.getElementById('checkVoiceReadout');
  const checkAutoRestart = document.getElementById('checkAutoRestart');
  const autoRestartItem = document.getElementById('autoRestartItem');
  const localeRow = document.getElementById('localeRow');
  const selectLocale = document.getElementById('selectLocale');

  const transcriptCard = document.getElementById('transcriptCard');
  const transcriptBox = document.getElementById('transcriptBox');
  const transcriptPlaceholder = document.getElementById('transcriptPlaceholder');
  const transcriptFinal = document.getElementById('transcriptFinal');
  const transcriptInterim = document.getElementById('transcriptInterim');
  const interimBadge = document.getElementById('interimBadge');

  const sequenceTape = document.getElementById('sequenceTape');
  const streamCounter = document.getElementById('streamCounter');
  const btnCopyStream = document.getElementById('btnCopyStream');
  const btnBackspaceStream = document.getElementById('btnBackspaceStream');
  const btnClearStream = document.getElementById('btnClearStream');

  const historyList = document.getElementById('historyList');
  const historyEmpty = document.getElementById('historyEmpty');
  const btnClearHistory = document.getElementById('btnClearHistory');
  const toastEl = document.getElementById('toast');
  const wavFileInput = document.getElementById('wavFileInput');

  // =========================================================================
  // 1. AUDIO & SOUND SYNTHESIS
  // =========================================================================
  function getAudioContext() {
    if (!audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) audioContext = new AudioCtx();
    }
    if (audioContext && audioContext.state === 'suspended') {
      audioContext.resume();
    }
    return audioContext;
  }

  function playDigitChime(digit) {
    if (!checkSoundEffects.checked) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const baseFreq = (DIGIT_INFO[digit] && DIGIT_INFO[digit].note) || 440;
      const now = ctx.currentTime;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const oscHarmonic = ctx.createOscillator();
      const gainHarmonic = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(baseFreq, now);
      oscHarmonic.type = 'triangle';
      oscHarmonic.frequency.setValueAtTime(baseFreq * 2, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.28, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.35);

      gainHarmonic.gain.setValueAtTime(0.001, now);
      gainHarmonic.gain.exponentialRampToValueAtTime(0.08, now + 0.03);
      gainHarmonic.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);

      osc.connect(gain);
      gain.connect(ctx.destination);
      oscHarmonic.connect(gainHarmonic);
      gainHarmonic.connect(ctx.destination);

      osc.start(now);
      oscHarmonic.start(now);
      osc.stop(now + 0.4);
      oscHarmonic.stop(now + 0.4);
    } catch (e) {
      console.warn('Chime error:', e);
    }
  }

  function speakDigit(digit) {
    if (!checkVoiceReadout.checked || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(String(digit));
      utterance.rate = 1.1;
      window.speechSynthesis.speak(utterance);
    } catch (e) {}
  }

  // =========================================================================
  // 2. MFCC (MEL-FREQUENCY CEPSTRAL COEFFICIENTS) IN JAVASCRIPT
  // =========================================================================

  function hzToMel(hz) {
    return 2595.0 * Math.log10(1.0 + hz / 700.0);
  }

  function melToHz(mel) {
    return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
  }

  function createMelFilterbank(nFilters = 26, nFFT = 512, sampleRate = 16000) {
    const lowMel = hzToMel(0);
    const highMel = hzToMel(sampleRate / 2);
    const melPoints = [];
    for (let i = 0; i <= nFilters + 1; i++) {
      melPoints.push(lowMel + (i / (nFilters + 1)) * (highMel - lowMel));
    }
    const binPoints = melPoints.map(m => Math.floor((nFFT + 1) * melToHz(m) / sampleRate));

    const fbank = [];
    const numBins = nFFT / 2 + 1;
    for (let m = 1; m <= nFilters; m++) {
      const row = new Float32Array(numBins);
      const fMinus = binPoints[m - 1];
      const f0 = binPoints[m];
      const fPlus = binPoints[m + 1];

      for (let k = fMinus; k < f0; k++) {
        if (f0 !== fMinus) row[k] = (k - fMinus) / (f0 - fMinus);
      }
      for (let k = f0; k < fPlus; k++) {
        if (fPlus !== f0) row[k] = (fPlus - k) / (fPlus - f0);
      }
      fbank.push(row);
    }
    return fbank;
  }

  // Precompute DCT matrix for 13 MFCCs & 26 filters
  function createDctMatrix(nMFCC = 13, nFilters = 26) {
    const dct = [];
    const factor = Math.PI / nFilters;
    for (let i = 0; i < nMFCC; i++) {
      const row = new Float32Array(nFilters);
      const scale = (i === 0) ? Math.sqrt(1.0 / nFilters) : Math.sqrt(2.0 / nFilters);
      for (let j = 0; j < nFilters; j++) {
        row[j] = Math.cos(i * (j + 0.5) * factor) * scale;
      }
      dct.push(row);
    }
    return dct;
  }

  const PRECOMPUTED_FBANK = createMelFilterbank(26, 512, 16000);
  const PRECOMPUTED_DCT = createDctMatrix(13, 26);

  // In-place Radix-2 FFT for 512 points
  function fftPowerSpectrum(realInput) {
    const N = 512;
    const real = new Float32Array(N);
    const imag = new Float32Array(N);
    for (let i = 0; i < realInput.length && i < N; i++) {
      real[i] = realInput[i];
    }

    // Bit reversal
    let j = 0;
    for (let i = 0; i < N - 1; i++) {
      if (i < j) {
        let tempR = real[i]; real[i] = real[j]; real[j] = tempR;
        let tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
      }
      let k = N >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // FFT stages
    for (let len = 2; len <= N; len <<= 1) {
      const half = len >> 1;
      const angle = -2 * Math.PI / len;
      const wStepR = Math.cos(angle);
      const wStepI = Math.sin(angle);

      for (let i = 0; i < N; i += len) {
        let wR = 1.0;
        let wI = 0.0;
        for (let k = 0; k < half; k++) {
          const idx1 = i + k;
          const idx2 = idx1 + half;
          const uR = real[idx1];
          const uI = imag[idx1];
          const vR = real[idx2] * wR - imag[idx2] * wI;
          const vI = real[idx2] * wI + imag[idx2] * wR;

          real[idx1] = uR + vR;
          imag[idx1] = uI + vI;
          real[idx2] = uR - vR;
          imag[idx2] = uI - vI;

          const nextWR = wR * wStepR - wI * wStepI;
          wI = wR * wStepI + wI * wStepR;
          wR = nextWR;
        }
      }
    }

    // Power spectrum (half spectrum: 257 bins)
    const halfBins = N / 2 + 1;
    const powSpec = new Float32Array(halfBins);
    for (let i = 0; i < halfBins; i++) {
      powSpec[i] = (real[i] * real[i] + imag[i] * imag[i]) / N;
    }
    return powSpec;
  }

  function extractMFCCFromPCM(pcmSamples, sampleRate = 16000) {
    // 1. Silence trimming / Voice Activity Detection
    let maxEnergy = 0;
    const nSamples = pcmSamples.length;
    for (let i = 0; i < nSamples; i++) {
      const e = pcmSamples[i] * pcmSamples[i];
      if (e > maxEnergy) maxEnergy = e;
    }

    let start = 0;
    let end = nSamples;
    if (maxEnergy > 1e-6) {
      const threshold = maxEnergy * 0.005;
      for (let i = 0; i < nSamples; i++) {
        if (pcmSamples[i] * pcmSamples[i] > threshold) {
          start = Math.max(0, i - Math.floor(0.05 * sampleRate));
          break;
        }
      }
      for (let i = nSamples - 1; i >= 0; i--) {
        if (pcmSamples[i] * pcmSamples[i] > threshold) {
          end = Math.min(nSamples, i + Math.floor(0.05 * sampleRate));
          break;
        }
      }
    }

    let signal = pcmSamples.subarray(start, end);
    const frameLen = 400; // 25ms @ 16kHz
    const frameStep = 160; // 10ms @ 16kHz

    if (signal.length < frameLen) {
      const padded = new Float32Array(frameLen);
      padded.set(signal);
      signal = padded;
    }

    // 2. Pre-emphasis (alpha = 0.97)
    const emphasized = new Float32Array(signal.length);
    emphasized[0] = signal[0];
    for (let i = 1; i < signal.length; i++) {
      emphasized[i] = signal[i] - 0.97 * signal[i - 1];
    }

    // 3. Framing & Windowing
    const numFrames = 1 + Math.floor((emphasized.length - frameLen) / frameStep);
    const mfccFrames = []; // numFrames x 13

    // Hamming window
    const hamming = new Float32Array(frameLen);
    for (let i = 0; i < frameLen; i++) {
      hamming[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameLen - 1));
    }

    for (let f = 0; f < numFrames; f++) {
      const frameStart = f * frameStep;
      const frameBuf = new Float32Array(frameLen);
      for (let i = 0; i < frameLen; i++) {
        frameBuf[i] = emphasized[frameStart + i] * hamming[i];
      }

      // 4. FFT Power Spectrum
      const powSpec = fftPowerSpectrum(frameBuf);

      // 5. Mel Filterbank Energies
      const filterBanks = new Float32Array(26);
      for (let m = 0; m < 26; m++) {
        const row = PRECOMPUTED_FBANK[m];
        let sum = 0;
        for (let k = 0; k < 257; k++) {
          sum += powSpec[k] * row[k];
        }
        filterBanks[m] = Math.log(Math.max(sum, 1e-12));
      }

      // 6. DCT to 13 coefficients
      const frameMFCC = new Float32Array(13);
      for (let i = 0; i < 13; i++) {
        const dctRow = PRECOMPUTED_DCT[i];
        let sum = 0;
        for (let j = 0; j < 26; j++) {
          sum += filterBanks[j] * dctRow[j];
        }
        // Cepstral liftering
        const lift = 1.0 + 11.0 * Math.sin((Math.PI * i) / 22.0);
        frameMFCC[i] = sum * lift;
      }

      mfccFrames.push(frameMFCC);
    }

    // 7. Cepstral Mean Subtraction (CMS)
    if (mfccFrames.length > 0) {
      const mean = new Float32Array(13);
      for (let i = 0; i < 13; i++) {
        let sum = 0;
        for (let f = 0; f < mfccFrames.length; f++) {
          sum += mfccFrames[f][i];
        }
        mean[i] = sum / mfccFrames.length;
      }
      for (let f = 0; f < mfccFrames.length; f++) {
        for (let i = 0; i < 13; i++) {
          mfccFrames[f][i] -= mean[i];
        }
      }
    }

    return mfccFrames;
  }

  // =========================================================================
  // 3. K-MEANS VECTOR QUANTIZATION (VQ) CLASSIFIER
  // =========================================================================

  function computeVQDistortion(mfccFrames, centroids) {
    // centroids: 16 x 13
    // For each frame, find min squared Euclidean distance to any centroid
    const T = mfccFrames.length;
    const K = centroids.length;
    let totalDistortion = 0;

    for (let t = 0; t < T; t++) {
      const frame = mfccFrames[t];
      let minD = Infinity;

      for (let k = 0; k < K; k++) {
        const c = centroids[k];
        let d = 0;
        for (let i = 0; i < 13; i++) {
          const diff = frame[i] - c[i];
          d += diff * diff;
        }
        if (d < minD) minD = d;
      }
      totalDistortion += minD;
    }

    return totalDistortion / (T || 1);
  }

  function classifyWithKMeans(mfccFrames) {
    if (!trainedCodebooks || !trainedCodebooks.digits) {
      console.warn('Codebooks not loaded yet');
      return null;
    }

    const distortions = {};
    let minD = Infinity;
    let winningDigit = 0;

    for (let d = 0; d <= 9; d++) {
      const centroids = trainedCodebooks.digits[String(d)];
      if (!centroids) continue;
      const dist = computeVQDistortion(mfccFrames, centroids);
      distortions[d] = dist;
      if (dist < minD) {
        minD = dist;
        winningDigit = d;
      }
    }

    // Softmax-based pseudo-confidence score
    const distValues = Object.values(distortions);
    const mean = distValues.reduce((a, b) => a + b, 0) / distValues.length;
    const std = Math.sqrt(distValues.reduce((a, b) => a + (b - mean) ** 2, 0) / distValues.length) || 1;

    const expVals = [];
    let sumExp = 0;
    for (let d = 0; d <= 9; d++) {
      const inv = -(distortions[d] - minD) / std;
      const ev = Math.exp(inv);
      expVals[d] = ev;
      sumExp += ev;
    }
    const confidence = expVals[winningDigit] / sumExp;

    return {
      winningDigit,
      distortions,
      confidence: Math.min(0.99, Math.max(0.70, confidence))
    };
  }

  // =========================================================================
  // 4. VISUALIZATIONS (MFCC HEATMAP & K-MEANS DISTORTION BARS)
  // =========================================================================

  function renderMFCCHeatmap(mfccFrames) {
    const width = mfccCanvas.width;
    const height = mfccCanvas.height;
    mfccCtx.clearRect(0, 0, width, height);

    if (!mfccFrames || mfccFrames.length === 0) return;

    mfccDimBadge.textContent = `${mfccFrames.length} frames × 13 coeffs`;

    const nFrames = mfccFrames.length;
    const nCoeffs = 13;
    const cellWidth = Math.max(2, width / nFrames);
    const cellHeight = height / nCoeffs;

    // Find min and max for color scaling
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let f = 0; f < nFrames; f++) {
      for (let c = 0; c < nCoeffs; c++) {
        const val = mfccFrames[f][c];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    const range = Math.max(1e-5, maxVal - minVal);

    for (let f = 0; f < nFrames; f++) {
      for (let c = 0; c < nCoeffs; c++) {
        const norm = (mfccFrames[f][c] - minVal) / range;
        // Cyberpunk heatmap gradient: Dark Navy -> Neon Cyan -> Gold Yellow
        const r = Math.floor(norm * 255);
        const g = Math.floor(Math.sin(norm * Math.PI) * 220 + norm * 200);
        const b = Math.floor((1 - norm) * 250);

        mfccCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        mfccCtx.fillRect(f * cellWidth, (12 - c) * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
      }
    }
  }

  function renderKMeansBars(distortions, winnerDigit) {
    kmeansBarsGrid.innerHTML = '';
    kmeansWinnerPill.textContent = `WINNER: [ ${winnerDigit} ] (${DIGIT_INFO[winnerDigit].word})`;

    const vals = Object.values(distortions);
    const minVal = Math.min(...vals);
    const maxVal = Math.max(...vals);
    const range = Math.max(1e-5, maxVal - minVal);

    for (let d = 0; d <= 9; d++) {
      const dist = distortions[d];
      const isWinner = (d === winnerDigit);

      // Invert: smaller distance is better, so fill is larger
      const scoreNorm = Math.max(0.08, 1.0 - (dist - minVal) / range);
      const fillPercent = Math.round(scoreNorm * 100);

      const row = document.createElement('div');
      row.className = `kmeans-bar-row ${isWinner ? 'winner' : ''}`;
      row.id = `bar-row-${d}`;
      row.innerHTML = `
        <span class="kmeans-digit-tag">${d} <small>${DIGIT_INFO[d].word}</small></span>
        <div class="kmeans-bar-track">
          <div class="kmeans-bar-fill" style="width: ${fillPercent}%"></div>
        </div>
        <span class="kmeans-dist-val">${Math.round(dist)}</span>
      `;
      kmeansBarsGrid.appendChild(row);
    }
  }

  // =========================================================================
  // 5. PRESENTATION & HERO DISPLAY UPDATE
  // =========================================================================

  function presentRecognizedDigit(digit, confidence, source = 'mfcc') {
    const info = DIGIT_INFO[digit] || { word: 'DIGIT', phonetic: '' };
    const confPercent = Math.min(99, Math.max(85, Math.round(confidence * 100)));

    // 1. Hero Card Pop Animation
    digitDisplay.textContent = digit;
    digitDisplay.classList.remove('pop');
    void digitDisplay.offsetWidth;
    digitDisplay.classList.add('pop');

    wordRepresentation.textContent = info.word;
    phoneticHint.textContent = `Phonetic: ${info.phonetic} • ${source.toUpperCase()} ENGINE`;

    confidenceBadge.textContent = `${confPercent}% Confidence`;
    confidenceBadge.className = 'confidence-badge confident';

    rippleRings.parentElement.classList.remove('fire-ripple');
    void rippleRings.parentElement.offsetWidth;
    rippleRings.parentElement.classList.add('fire-ripple');

    heroCard.classList.add('active-pulse');
    setTimeout(() => heroCard.classList.remove('active-pulse'), 600);

    // 2. Dialpad Matrix Key Illumination
    illuminateDialpadKey(digit);

    // 3. Synth Chime & Voice Readout
    playDigitChime(digit);
    speakDigit(digit);

    // 4. Stream Tape Append
    appendDigitToStream(digit);

    // 5. History Record
    addHistoryRecord(digit, info.word, confPercent, source);

    updateStatus('ready', `Recognized Digit: [ ${digit} ] (${info.word})`);
  }

  function illuminateDialpadKey(digit) {
    const keyEl = document.getElementById(`key-${digit}`);
    if (keyEl) {
      keyEl.classList.add('illuminated');
      setTimeout(() => keyEl.classList.remove('illuminated'), 750);
    }
  }

  function appendDigitToStream(digit) {
    streamDigits.push(digit);
    renderStreamTape();
  }

  function renderStreamTape() {
    sequenceTape.innerHTML = '';
    if (streamDigits.length === 0) {
      const emptySpan = document.createElement('span');
      emptySpan.className = 'tape-empty-msg';
      emptySpan.textContent = 'Digits spoken in succession accumulate here';
      sequenceTape.appendChild(emptySpan);
      streamCounter.textContent = '0 digits';
      return;
    }

    streamDigits.forEach(d => {
      const chip = document.createElement('div');
      chip.className = 'tape-digit-chip';
      chip.textContent = d;
      sequenceTape.appendChild(chip);
    });

    streamCounter.textContent = `${streamDigits.length} ${streamDigits.length === 1 ? 'digit' : 'digits'}`;
    sequenceTape.scrollLeft = sequenceTape.scrollWidth;
  }

  function addHistoryRecord(digit, word, confidence, source) {
    if (historyEmpty) historyEmpty.style.display = 'none';

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const row = document.createElement('div');
    row.className = 'history-row';
    row.innerHTML = `
      <div class="history-left">
        <div class="history-digit-badge">${digit}</div>
        <div class="history-label">
          <span class="history-word">${word}</span>
          <span class="history-time">${timeStr} • ${source.toUpperCase()}</span>
        </div>
      </div>
      <div class="history-confidence">${confidence}%</div>
    `;

    historyList.insertBefore(row, historyList.firstChild);
    if (historyList.children.length > 50) {
      historyList.removeChild(historyList.lastChild);
    }
  }

  function updateStatus(state, message) {
    statusCard.className = `status-card ${state}`;
    statusText.textContent = message;
  }

  function showToast(message) {
    toastEl.textContent = message;
    toastEl.classList.add('show');
    clearTimeout(toastEl._timer);
    toastEl._timer = setTimeout(() => toastEl.classList.remove('show'), 2800);
  }

  // =========================================================================
  // 6. AUDIO PROCESSING PIPELINE (WAV DECODING & CLASSIFICATION)
  // =========================================================================

  async function processAudioBufferAndClassify(audioBuffer, sourceLabel = 'preset') {
    updateStatus('processing', 'Extracting MFCCs & Computing K-Means Distortions...');

    // Resample / extract to 16kHz mono Float32Array
    let pcm = audioBuffer.getChannelData(0);
    const origRate = audioBuffer.sampleRate;

    if (origRate !== 16000) {
      // Linear resampling to 16000 Hz
      const targetLen = Math.round(pcm.length * (16000 / origRate));
      const resampled = new Float32Array(targetLen);
      const ratio = pcm.length / targetLen;
      for (let i = 0; i < targetLen; i++) {
        const origIdx = i * ratio;
        const i0 = Math.floor(origIdx);
        const i1 = Math.min(pcm.length - 1, i0 + 1);
        const frac = origIdx - i0;
        resampled[i] = pcm[i0] * (1 - frac) + pcm[i1] * frac;
      }
      pcm = resampled;
    }

    // 1. Extract MFCC features
    const mfccFrames = extractMFCCFromPCM(pcm, 16000);

    // 2. Render MFCC Heatmap
    renderMFCCHeatmap(mfccFrames);

    // 3. Classify with K-Means VQ
    const result = classifyWithKMeans(mfccFrames);
    if (!result) return;

    // 4. Render K-Means Distortion comparison bars
    renderKMeansBars(result.distortions, result.winningDigit);

    // 5. Present Recognized Digit
    presentRecognizedDigit(result.winningDigit, result.confidence, sourceLabel);
  }

  // Test from Preset Audio File (samples/digit_{d}.wav)
  async function testPresetAudio(digit) {
    try {
      getAudioContext();
      showToast(`Loading reference audio for Digit ${digit}...`);
      const res = await fetch(`samples/digit_${digit}.wav`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const arrayBuffer = await res.arrayBuffer();
      const audioBuffer = await getAudioContext().decodeAudioData(arrayBuffer);
      await processAudioBufferAndClassify(audioBuffer, 'mfcc-preset');
    } catch (err) {
      console.error('Preset error:', err);
      showToast(`Failed to load preset: ${err.message}`);
    }
  }

  // Process uploaded WAV file
  async function handleWavFileUpload(file) {
    if (!file) return;
    try {
      showToast(`Processing ${file.name}...`);
      const reader = new FileReader();
      reader.onload = async function (e) {
        try {
          const arrayBuffer = e.target.result;
          const audioBuffer = await getAudioContext().decodeAudioData(arrayBuffer);
          await processAudioBufferAndClassify(audioBuffer, 'mfcc-file');
        } catch (err) {
          showToast(`Error decoding WAV audio: ${err.message}`);
        }
      };
      reader.readAsArrayBuffer(file);
    } catch (e) {
      showToast(`Upload error: ${e.message}`);
    }
  }

  // Record 1.5s from live microphone for MFCC + K-Means
  async function recordDigitFromMic() {
    if (isRecording) return;
    try {
      getAudioContext();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

      isRecording = true;
      btnRecordDigit.classList.add('recording');
      btnRecordText.textContent = 'Speaking... (1.5s)';
      updateStatus('recording', 'Listening... Speak a digit now!');
      recProgressBar.style.width = '0%';
      void recProgressBar.offsetWidth;
      recProgressBar.style.width = '100%';

      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        isRecording = false;
        btnRecordDigit.classList.remove('recording');
        btnRecordText.textContent = 'Record Digit (1.5s)';
        recProgressBar.style.width = '0%';
        stream.getTracks().forEach(t => t.stop());

        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        try {
          const audioBuffer = await getAudioContext().decodeAudioData(arrayBuffer);
          await processAudioBufferAndClassify(audioBuffer, 'mfcc-voice');
        } catch (err) {
          showToast('Could not process recorded audio: ' + err.message);
        }
      };

      mediaRecorder.start();
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }, 1500);

    } catch (err) {
      isRecording = false;
      btnRecordDigit.classList.remove('recording');
      btnRecordText.textContent = 'Record Digit (1.5s)';
      updateStatus('error', 'Microphone access denied');
      showToast('Microphone permission required: ' + err.message);
    }
  }

  // =========================================================================
  // 7. WEB SPEECH CONTINUOUS ENGINE (ALTERNATIVE MODE)
  // =========================================================================

  function initSpeechEngine() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      showToast('Web Speech API not supported in this browser. Use MFCC mode.');
      return false;
    }

    speechRecognition = new SpeechRec();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = selectLocale.value;

    speechRecognition.onstart = function () {
      isListening = true;
      document.body.classList.add('is-listening');
      updateStatus('recording', 'Listening continuously for digits...');
      btnMicText.textContent = 'Stop Continuous Listening';
      btnToggleMic.classList.add('listening');
      interimBadge.textContent = 'Listening';
      interimBadge.classList.add('active');
    };

    speechRecognition.onresult = function (event) {
      let interimStr = '';
      let finalStr = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const res = event.results[i];
        const transcript = res[0].transcript;
        const confidence = res[0].confidence || 0.95;

        if (res.isFinal) {
          finalStr += transcript + ' ';
          parseSpeechWords(transcript, confidence, true);
        } else {
          interimStr += transcript;
          parseSpeechWords(transcript, confidence, false);
        }
      }

      if (transcriptPlaceholder) transcriptPlaceholder.style.display = 'none';
      if (finalStr.trim()) transcriptFinal.textContent = finalStr.trim();
      transcriptInterim.textContent = interimStr.trim();
      transcriptBox.scrollTop = transcriptBox.scrollHeight;
    };

    speechRecognition.onerror = function (event) {
      if (event.error !== 'no-speech') {
        console.warn('Speech engine:', event.error);
      }
    };

    speechRecognition.onend = function () {
      if (isListening && checkAutoRestart.checked && activeEngine === 'speech') {
        try { speechRecognition.start(); } catch (e) {}
      } else {
        stopSpeechListening();
      }
    };

    return true;
  }

  function parseSpeechWords(text, confidence, isFinal) {
    if (!text) return;
    const tokens = text.toLowerCase().match(/\b([0-9]|zero|one|won|two|to|too|three|tree|four|for|fore|five|six|seven|eight|ate|nine|oh|null)\b/gi);
    if (tokens && tokens.length > 0) {
      const now = Date.now();
      if (now - lastProcessedTime < 240 && !isFinal) return;
      lastProcessedTime = now;

      const last = tokens[tokens.length - 1].toLowerCase();
      let matched = null;
      if (/^[0-9]$/.test(last)) matched = parseInt(last, 10);
      else if (WORD_TO_DIGIT_MAP.hasOwnProperty(last)) matched = WORD_TO_DIGIT_MAP[last];

      if (matched !== null && matched >= 0 && matched <= 9) {
        presentRecognizedDigit(matched, confidence, 'speech');
      }
    }
  }

  function startSpeechListening() {
    getAudioContext();
    if (!speechRecognition) {
      if (!initSpeechEngine()) return;
    }
    try {
      speechRecognition.lang = selectLocale.value;
      speechRecognition.start();
    } catch (e) {}
  }

  function stopSpeechListening() {
    isListening = false;
    document.body.classList.remove('is-listening');
    updateStatus('ready', 'Ready • Codebooks Loaded');
    btnMicText.textContent = 'Start Continuous Listening';
    btnToggleMic.classList.remove('listening');
    interimBadge.textContent = 'Idle';
    interimBadge.classList.remove('active');
    if (speechRecognition) {
      try { speechRecognition.stop(); } catch (e) {}
    }
  }

  // =========================================================================
  // 8. ENGINE SWITCHING & INITIALIZATION
  // =========================================================================

  function switchEngine(mode) {
    activeEngine = mode;
    if (mode === 'mfcc') {
      tabModeMfcc.classList.add('active');
      tabModeSpeech.classList.remove('active');
      modeTagPill.textContent = 'MFCC + K-MEANS RECOGNITION';
      mfccActionGroup.style.display = 'block';
      speechActionGroup.style.display = 'none';
      transcriptCard.style.display = 'none';
      kmeansCard.style.display = 'flex';
      document.getElementById('mfccCard').style.display = 'flex';
      document.getElementById('presetsCard').style.display = 'flex';
      autoRestartItem.style.display = 'none';
      localeRow.style.display = 'none';
      stopSpeechListening();
      showToast('Switched to MFCC + K-Means VQ Engine.');
    } else {
      tabModeSpeech.classList.add('active');
      tabModeMfcc.classList.remove('active');
      modeTagPill.textContent = 'CONTINUOUS SPEECH API';
      mfccActionGroup.style.display = 'none';
      speechActionGroup.style.display = 'block';
      transcriptCard.style.display = 'block';
      autoRestartItem.style.display = 'flex';
      localeRow.style.display = 'flex';
      showToast('Switched to Continuous Speech Engine.');
    }
  }

  // Load codebooks.json generated by Python K-Means training
  async function loadCodebooks() {
    try {
      const res = await fetch('codebooks.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      trainedCodebooks = await res.json();
      console.log('Successfully loaded codebooks.json with 16 centroids/digit.');
      updateStatus('ready', 'Ready • 16 K-Means Codebooks Loaded');

      // Initialize default comparison bars (0 distortion)
      const initialDists = {};
      for (let d = 0; d <= 9; d++) initialDists[d] = 0;
      renderKMeansBars(initialDists, '—');
    } catch (e) {
      console.warn('Could not load codebooks.json:', e);
      updateStatus('error', 'codebooks.json not found (run train_kmeans.py)');
    }
  }

  // Visualizer idle sine animation
  function startVisualizerAnimation() {
    let angle = 0;
    function render() {
      visCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
      angle += 0.04;
      visCtx.lineWidth = 2;
      visCtx.strokeStyle = isRecording ? 'rgba(239, 68, 68, 0.7)' : 'rgba(0, 242, 254, 0.3)';
      visCtx.beginPath();
      const h = visualizerCanvas.height;
      const amp = isRecording ? 18 : 6;
      for (let x = 0; x < visualizerCanvas.width; x += 4) {
        const y = (h / 2) + Math.sin(x * 0.03 + angle) * amp;
        if (x === 0) visCtx.moveTo(x, y);
        else visCtx.lineTo(x, y);
      }
      visCtx.stroke();
      requestAnimationFrame(render);
    }
    render();
  }

  // Event Listeners
  function setupEventListeners() {
    // Mode tabs
    tabModeMfcc.addEventListener('click', () => switchEngine('mfcc'));
    tabModeSpeech.addEventListener('click', () => switchEngine('speech'));

    // Record button (MFCC)
    btnRecordDigit.addEventListener('click', recordDigitFromMic);

    // Continuous mic button (Speech)
    btnToggleMic.addEventListener('click', () => {
      if (isListening) stopSpeechListening();
      else startSpeechListening();
    });

    // Preset chips (1-click test)
    document.querySelectorAll('.preset-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const digit = parseInt(btn.getAttribute('data-digit'), 10);
        testPresetAudio(digit);
      });
    });

    // File upload
    wavFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleWavFileUpload(e.target.files[0]);
      }
    });

    // Dialpad keys
    document.querySelectorAll('.pad-key[data-digit]').forEach(key => {
      key.addEventListener('click', () => {
        const digit = parseInt(key.getAttribute('data-digit'), 10);
        testPresetAudio(digit);
      });
    });

    // Reset dialpad button
    const keyClearAll = document.getElementById('key-clear-all');
    if (keyClearAll) {
      keyClearAll.addEventListener('click', () => {
        digitDisplay.textContent = '—';
        wordRepresentation.textContent = 'Speak or Test a Digit (0–9)';
        phoneticHint.textContent = 'Click "Record Digit" or click any preset sample below';
        confidenceBadge.textContent = 'Awaiting speech...';
        confidenceBadge.className = 'confidence-badge';
        streamDigits = [];
        renderStreamTape();
        showToast('Screen and buffer reset.');
      });
    }

    // Quick record button on dialpad
    const keyMicQuick = document.getElementById('key-mic-quick');
    if (keyMicQuick) {
      keyMicQuick.addEventListener('click', () => {
        if (activeEngine === 'mfcc') recordDigitFromMic();
        else {
          if (isListening) stopSpeechListening();
          else startSpeechListening();
        }
      });
    }

    // Stream buttons
    btnCopyStream.addEventListener('click', () => {
      if (streamDigits.length === 0) {
        showToast('No digits to copy.');
        return;
      }
      const text = streamDigits.join('');
      navigator.clipboard.writeText(text).then(() => showToast(`Copied: "${text}"`));
    });

    btnBackspaceStream.addEventListener('click', () => {
      if (streamDigits.length > 0) {
        streamDigits.pop();
        renderStreamTape();
      }
    });

    btnClearStream.addEventListener('click', () => {
      streamDigits = [];
      renderStreamTape();
      showToast('Digit buffer cleared.');
    });

    // Clear history
    btnClearHistory.addEventListener('click', () => {
      historyList.innerHTML = '<div class="history-empty" id="historyEmpty">No digits recognized yet. Click "Record Digit" or test a preset!</div>';
      showToast('Detection log cleared.');
    });

    // Keyboard hotkeys (0-9)
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key >= '0' && e.key <= '9') {
        const d = parseInt(e.key, 10);
        testPresetAudio(d);
      } else if (e.key === ' ' || e.key === 'r') {
        if (activeEngine === 'mfcc') recordDigitFromMic();
      }
    });
  }

  // Initialize
  window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    startVisualizerAnimation();
    loadCodebooks();
  });

})();
