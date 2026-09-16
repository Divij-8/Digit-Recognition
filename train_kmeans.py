"""
Dataset Generator & K-Means Codebook Training Script
Synthesizes spoken digit audio on macOS, extracts MFCCs, trains K-Means codebooks,
evaluates accuracy, and exports codebooks to codebooks.json.
"""

import os
import subprocess
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mfcc import read_wav, extract_mfcc
from kmeans import DigitVQClassifier

DIGIT_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine"
}

# Diverse natural macOS speech voices across male, female, and varied accents
TRAIN_VOICES = ["Samantha", "Fred", "Albert", "Karen", "Daniel", "Kathy", "Moira", "Rishi", "Tessa", "Aman"]
TEST_VOICES = ["Daniel", "Karen", "Rishi"]
RATES = [160, 185, 210]

def get_available_voices(candidate_voices):
    """Filters available voices on this macOS machine."""
    try:
        out = subprocess.check_output(["say", "-v", "?"], text=True)
        installed = [line.split()[0] for line in out.strip().split("\n") if line]
        avail = [v for v in candidate_voices if v in installed]
        if not avail:
            return ["Alex"] if "Alex" in installed else [installed[0]]
        return avail
    except Exception:
        return ["Alex"]

def _generate_single_wav(args):
    voice, rate, word, fname = args
    if not os.path.exists(fname) or os.path.getsize(fname) == 0:
        cmd = [
            "say", "-v", voice, "-r", str(rate),
            word, "-o", fname, "--data-format=LEI16@16000"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fname

def synthesize_dataset(base_dir="dataset"):
    """
    Synthesizes WAV files (16kHz mono) in parallel for speed.
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    available_train = get_available_voices(TRAIN_VOICES)
    available_test = get_available_voices(TEST_VOICES)
    if not available_test:
        available_test = available_train[:1]

    print(f"Generating training audio using voices: {available_train}", flush=True)
    print(f"Generating test audio using voices: {available_test}", flush=True)

    tasks = []
    train_files = {d: [] for d in range(10)}
    for digit, word in DIGIT_WORDS.items():
        for voice in available_train:
            for rate in RATES:
                fname = os.path.join(train_dir, f"digit_{digit}_{voice}_r{rate}.wav")
                train_files[digit].append(fname)
                tasks.append((voice, rate, word, fname))

    test_files = {d: [] for d in range(10)}
    for digit, word in DIGIT_WORDS.items():
        for voice in available_test:
            for rate in [175, 195]:
                fname = os.path.join(test_dir, f"digit_{digit}_{voice}_r{rate}.wav")
                test_files[digit].append(fname)
                tasks.append((voice, rate, word, fname))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_generate_single_wav, tasks))

    print(f"Dataset generated: {len(tasks)} audio files ready.", flush=True)
    return train_files, test_files

def train_and_evaluate():
    print("=" * 60, flush=True)
    print("STEP 1: Synthesizing Audio Dataset (0–9)", flush=True)
    print("=" * 60, flush=True)
    train_files, test_files = synthesize_dataset()

    classifier = DigitVQClassifier(n_clusters_per_digit=16)

    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Extracting MFCCs & Training K-Means Codebooks", flush=True)
    print("=" * 60, flush=True)

    for digit in range(10):
        pooled_frames = []
        files = train_files[digit]
        for f in files:
            signal, sr = read_wav(f)
            mfcc_feats = extract_mfcc(signal, sample_rate=sr)
            pooled_frames.append(mfcc_feats)

        all_digit_mfcc = np.vstack(pooled_frames)
        classifier.train_digit(digit, all_digit_mfcc)
        print(f"Digit {digit} ('{DIGIT_WORDS[digit]}'): Trained codebook with 16 centroids across {all_digit_mfcc.shape[0]} frames.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Evaluating on Held-out Test Set", flush=True)
    print("=" * 60, flush=True)

    total_test = 0
    correct_test = 0

    for true_digit in range(10):
        for f in test_files[true_digit]:
            signal, sr = read_wav(f)
            test_mfcc = extract_mfcc(signal, sample_rate=sr)
            pred_digit, dists, conf = classifier.predict(test_mfcc)

            total_test += 1
            is_correct = (pred_digit == true_digit)
            if is_correct:
                correct_test += 1

    accuracy = (correct_test / total_test) * 100 if total_test > 0 else 0
    print(f"\nFinal Test Accuracy: {correct_test}/{total_test} ({accuracy:.1f}%)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Exporting Codebooks & Web Samples", flush=True)
    print("=" * 60, flush=True)

    export_path = "codebooks.json"
    with open(export_path, "w") as f:
        json.dump(classifier.to_dict(), f, indent=2)

    # Generate reference samples in samples/ for web application preset testing
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    for digit, word in DIGIT_WORDS.items():
        sample_path = os.path.join(samples_dir, f"digit_{digit}.wav")
        cmd = ["say", "-v", "Daniel", "-r", "175", word, "-o", sample_path, "--data-format=LEI16@16000"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Saved codebooks to {export_path} ({os.path.getsize(export_path)} bytes)", flush=True)
    print(f"Saved 10 reference audio samples in {samples_dir}/", flush=True)
    return classifier

if __name__ == "__main__":
    train_and_evaluate()
