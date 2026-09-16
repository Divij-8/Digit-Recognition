"""
CLI Voice Digit Recognition using MFCC & K-Means Codebooks
Usage:
    python3 recognize.py <path_to_wav_file>
    python3 recognize.py --test
"""

import sys
import os
import json
import numpy as np
from mfcc import read_wav, extract_mfcc
from kmeans import DigitVQClassifier

DIGIT_WORDS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]

def load_model(codebooks_path="codebooks.json"):
    if not os.path.exists(codebooks_path):
        print(f"Error: {codebooks_path} not found. Please run 'python3 train_kmeans.py' first.")
        sys.exit(1)
    with open(codebooks_path, "r") as f:
        data = json.load(f)
    classifier = DigitVQClassifier().from_dict(data)
    return classifier

def recognize_file(wav_path, classifier=None):
    if classifier is None:
        classifier = load_model()

    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return None

    signal, sr = read_wav(wav_path)
    feats = extract_mfcc(signal, sample_rate=sr)

    pred_digit, distortions, confidence = classifier.predict(feats)

    # Pretty output
    print("\n" + "=" * 62)
    print(f"  VOICE DIGIT RECOGNITION (MFCC + K-MEANS VQ)")
    print("=" * 62)
    print(f" Audio File    : {os.path.basename(wav_path)}")
    print(f" Sample Rate   : {sr} Hz ({len(signal)} samples, {len(signal)/sr:.2f}s)")
    print(f" MFCC Frames   : {feats.shape[0]} frames x {feats.shape[1]} coefficients")
    print("-" * 62)
    print(" Digit | Word   | Distortion (Avg Dist^2) | Relative Fit")
    print("-" * 62)

    min_dist = min(distortions.values())
    max_dist = max(distortions.values())
    rng = max(1e-5, max_dist - min_dist)

    for d in range(10):
        dist = distortions[d]
        is_winner = (d == pred_digit)
        marker = "▶ WINNER" if is_winner else "        "

        # Fit bar (shorter distance = longer green bar)
        score_norm = max(0.0, 1.0 - (dist - min_dist) / rng)
        bar_len = int(round(score_norm * 18))
        bar_str = "█" * bar_len + "░" * (18 - bar_len)

        print(f"   {d}   | {DIGIT_WORDS[d]:<6} |       {dist:>8.2f}          | [{bar_str}] {marker}")

    print("=" * 62)
    print(f"  >>> RECOGNIZED DIGIT: [ {pred_digit} ] ({DIGIT_WORDS[pred_digit]}) <<<")
    print(f"  Confidence Score   : {confidence * 100:.1f}%")
    print("=" * 62 + "\n")
    return pred_digit

def run_test_suite():
    classifier = load_model()
    test_dir = "dataset/test"
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found.")
        return

    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])
    if not files:
        print("No test files found in dataset/test.")
        return

    print(f"\nRunning evaluation on {len(files)} test samples...")
    correct = 0
    for f in files:
        # Extract ground truth digit from filename e.g. digit_7_Alex...
        parts = f.split("_")
        true_digit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1

        path = os.path.join(test_dir, f)
        signal, sr = read_wav(path)
        feats = extract_mfcc(signal, sample_rate=sr)
        pred, dists, conf = classifier.predict(feats)

        status = "✓ PASS" if pred == true_digit else "✗ FAIL"
        if pred == true_digit:
            correct += 1
        print(f"  {status} | File: {f:<30} | True: {true_digit} -> Pred: {pred} ({conf*100:.1f}%)")

    acc = (correct / len(files)) * 100
    print(f"\nTotal Accuracy: {correct}/{len(files)} ({acc:.1f}%)\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_suite()
    elif len(sys.argv) > 1:
        recognize_file(sys.argv[1])
    else:
        # If no arguments provided, test on first test file if available
        test_dir = "dataset/test"
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
            if files:
                recognize_file(os.path.join(test_dir, files[0]))
            else:
                print("Usage: python3 recognize.py <path_to_wav_file>")
        else:
            print("Usage: python3 recognize.py <path_to_wav_file>")
