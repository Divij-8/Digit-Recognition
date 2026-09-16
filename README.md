# 🎙 Voice Digit Recognition System (MFCC & K-Means)

A state-of-the-art Voice Digit Recognition system powered by **Mel-Frequency Cepstral Coefficients (MFCC)** for acoustic feature extraction and **K-Means Clustering** for Vector Quantization (VQ) codebook classification.

---

## ⚡ How It Works

### 1. Feature Extraction: MFCC
When a user speaks a digit, the audio signal ($16\text{ kHz}$ PCM) passes through an acoustic feature extraction pipeline:
1. **Voice Activity Detection (VAD)**: Trims leading and trailing silence based on short-time energy.
2. **Pre-emphasis**: High-pass filter $y[n] = x[n] - 0.97 \cdot x[n-1]$ to balance high-frequency formants.
3. **Framing & Windowing**: $25\text{ ms}$ frame length ($400$ samples), $10\text{ ms}$ hop ($160$ samples) multiplied by a **Hamming window**.
4. **FFT & Power Spectrum**: 512-point Fast Fourier Transform to extract spectral energy distribution.
5. **Mel Filterbank**: $26$ triangular bandpass filters spaced linearly below $1000\text{ Hz}$ and logarithmically above $1000\text{ Hz}$ according to the Mel scale:
   $$m = 2595 \log_{10}(1 + f / 700)$$
6. **Log Energy & DCT-II**: Discrete Cosine Transform compresses the $26$ filterbank energies into $13$ cepstral coefficients ($c_0 \dots c_{12}$).
7. **Cepstral Liftering & Mean Subtraction (CMS)**: Normalizes channel response and emphasizes formant information.

### 2. Classification: K-Means Vector Quantization (VQ)
For speech recognition, each digit class $d \in \{0, 1, \dots, 9\}$ has an individual codebook $C_d$ composed of $K=16$ cluster centroids in $\mathbb{R}^{13}$, trained using K-Means clustering over pooled multi-speaker training frames:
- Given an unknown spoken digit with $T$ MFCC frames $X = [x_1, x_2, \dots, x_T]$, we compute the **Quantization Distortion** to each digit's codebook:
  $$D(X, C_d) = \frac{1}{T} \sum_{t=1}^T \min_{c_k \in C_d} \|x_t - c_k\|^2$$
- The recognized digit is the one with the **minimum Euclidean distortion**:
  $$d^* = \arg\min_d D(X, C_d)$$

---

## 🚀 Quick Start

### 1. Launch the Interactive Web Application
A local web server is already running on port 8000:
```bash
# Open in your browser:
http://localhost:8000
```
*(Or manually run `python3 -m http.server 8000`)*

**Features in the Web App:**
- **MFCC + K-Means VQ Engine (Recommended)**:
  - Click **"Record Digit (1.5s)"** and speak any digit (0–9) into your microphone.
  - Or click any of the **1-Click Audio Preset Chips** (`0` to `9`) to test real pre-recorded reference voice samples.
  - Or upload your own custom `.wav` audio file.
  - **Live 2D MFCC Spectrogram**: Visualizes the 13 coefficients over time.
  - **K-Means Distortion Chart**: Displays real-time Euclidean distance bars across all digits $0\dots9$, highlighting the minimum distance as the WINNER.
  - **Glowing Hero Digit**: Shows the recognized digit with pop-in animations and synthesizer chimes.
- **Continuous Web Speech API Engine**:
  - Hands-free continuous listening mode that writes spoken digits to the screen in real-time.

---

### 2. Python CLI & Testing

#### Test a specific audio file:
```bash
python3 recognize.py dataset/test/digit_7_Daniel_r175.wav
```
Output:
```text
==============================================================
  VOICE DIGIT RECOGNITION (MFCC + K-MEANS VQ)
==============================================================
 Audio File    : digit_7_Daniel_r175.wav
 Sample Rate   : 16000 Hz (9707 samples, 0.61s)
 MFCC Frames   : 59 frames x 13 coefficients
--------------------------------------------------------------
 Digit | Word   | Distortion (Avg Dist^2) | Relative Fit
--------------------------------------------------------------
   0   | ZERO   |        3918.46          | [░░░░░░░░░░░░░░░░░░]         
   1   | ONE    |        3885.17          | [█░░░░░░░░░░░░░░░░░]         
   2   | TWO    |        3172.12          | [████████░░░░░░░░░░]         
   3   | THREE  |        3818.92          | [█░░░░░░░░░░░░░░░░░]         
   4   | FOUR   |        3526.53          | [████░░░░░░░░░░░░░░]         
   5   | FIVE   |        2950.36          | [██████████░░░░░░░░]         
   6   | SIX    |        3105.09          | [████████░░░░░░░░░░]         
   7   | SEVEN  |        2094.42          | [██████████████████] ▶ WINNER
   8   | EIGHT  |        2693.38          | [████████████░░░░░░]         
   9   | NINE   |        3968.37          | [░░░░░░░░░░░░░░░░░░]         
==============================================================
  >>> RECOGNIZED DIGIT: [ 7 ] (SEVEN) <<<
==============================================================
```

#### Run the full test suite:
```bash
python3 recognize.py --test
```
*(Evaluates 60 test audio samples across held-out test speakers Daniel, Karen, and Rishi: **100% accuracy**).*

#### Retrain K-Means Codebooks:
```bash
python3 train_kmeans.py
```
*(Synthesizes 360 audio samples across 10 natural macOS voices at varying speech rates, extracts MFCCs, and exports updated cluster centroids to `codebooks.json`).*

---

## 📁 Repository Structure

```text
├── index.html          # Interactive Web UI with MFCC canvas & K-Means chart
├── style.css           # Futuristic cyberpunk dark glassmorphism design
├── app.js              # In-browser MFCC extraction & K-Means VQ classifier
├── mfcc.py             # Pure NumPy MFCC feature extraction (zero broken dependencies)
├── kmeans.py           # Pure NumPy K-Means & DigitVQClassifier
├── train_kmeans.py     # Multi-speaker dataset synthesizer & codebook trainer
├── recognize.py        # Command-line recognition tool & test suite
├── codebooks.json      # Trained centroids (16 centroids × 13 dimensions per digit)
├── dataset/            # Training & testing WAV datasets
└── samples/            # 10 reference audio files (0–9) for instant browser testing
```
