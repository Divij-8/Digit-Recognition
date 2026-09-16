"""
Mel-Frequency Cepstral Coefficients (MFCC) Extraction Module
Implemented in pure NumPy for maximum portability and zero dependency conflicts.
"""

import numpy as np
import wave
import struct

def read_wav(filename):
    """
    Reads a standard WAV file and returns (audio_samples_float32, sample_rate).
    Normalized to range [-1.0, 1.0].
    """
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    # If stereo, average channels to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate

def hz_to_mel(hz):
    """Converts frequency in Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    """Converts Mel scale value to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_filter_banks(n_filters=26, n_fft=512, sample_rate=16000, low_freq=0, high_freq=None):
    """
    Constructs a triangular Mel filterbank.
    Returns matrix of shape (n_filters, n_fft // 2 + 1).
    """
    if high_freq is None:
        high_freq = sample_rate / 2.0

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, n_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    return fbank

def dct_matrix(n_mfcc=13, n_filters=26):
    """
    Computes standard Type-II DCT transformation matrix.
    Shape: (n_mfcc, n_filters).
    """
    dct_m = np.zeros((n_mfcc, n_filters), dtype=np.float32)
    factor = np.pi / n_filters
    for i in range(n_mfcc):
        for j in range(n_filters):
            dct_m[i, j] = np.cos(i * (j + 0.5) * factor)
        if i == 0:
            dct_m[i, :] *= np.sqrt(1.0 / n_filters)
        else:
            dct_m[i, :] *= np.sqrt(2.0 / n_filters)
    return dct_m

def extract_mfcc(signal, sample_rate=16000, frame_len_sec=0.025, frame_step_sec=0.010,
                 n_mfcc=13, n_filters=26, n_fft=512, preemph=0.97, ceplifter=22):
    """
    Extracts MFCC features from raw 1D audio signal.
    Returns:
      mfcc_features: ndarray of shape (n_frames, n_mfcc)
    """
    if signal is None or len(signal) == 0:
        signal = np.zeros(int(frame_len_sec * sample_rate), dtype=np.float32)

    # 1. Voice Activity Detection / Silence Trimming based on energy
    energy = signal ** 2
    max_energy = np.max(energy) if len(energy) > 0 else 0.0
    if max_energy > 1e-8:
        threshold = max_energy * 0.005
        active_indices = np.where(energy > threshold)[0]
        if len(active_indices) > 0:
            start_idx = max(0, active_indices[0] - int(0.05 * sample_rate))
            end_idx = min(len(signal), active_indices[-1] + int(0.05 * sample_rate))
            signal = signal[start_idx:end_idx]

    if len(signal) < int(frame_len_sec * sample_rate):
        # Pad short signals
        signal = np.pad(signal, (0, int(frame_len_sec * sample_rate) - len(signal)))

    # 2. Pre-emphasis filter
    emphasized = np.append(signal[0], signal[1:] - preemph * signal[:-1])

    # 3. Framing
    frame_length = int(round(frame_len_sec * sample_rate))
    frame_step = int(round(frame_step_sec * sample_rate))
    signal_length = len(emphasized)

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 4. Windowing (Hamming)
    frames *= np.hamming(frame_length)

    # 5. FFT and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # 6. Mel Filterbank Energy
    fbank = get_filter_banks(n_filters=n_filters, n_fft=n_fft, sample_rate=sample_rate)
    filter_banks = np.dot(pow_frames, fbank.T)
    # Numerical stability
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fb = np.log(filter_banks)

    # 7. Discrete Cosine Transform (DCT)
    dct_m = dct_matrix(n_mfcc=n_mfcc, n_filters=n_filters)
    mfcc = np.dot(log_fb, dct_m.T)

    # 8. Cepstral Liftering
    if ceplifter > 0:
        n_coeff = mfcc.shape[1]
        n = np.arange(n_coeff)
        lift = 1.0 + (ceplifter / 2.0) * np.sin(np.pi * n / ceplifter)
        mfcc *= lift

    # 9. Cepstral Mean Subtraction (CMS)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc.astype(np.float32)

if __name__ == '__main__':
    # Quick sanity check with synthetic sine wave
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    feats = extract_mfcc(dummy_audio, sr)
    print(f"Sanity Check: Extracted MFCC shape: {feats.shape} (frames, coeffs)")
