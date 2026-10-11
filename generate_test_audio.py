#!/usr/bin/env python3
"""
Test Audio Generator - Creates various WAV files to test the Audio Analyzer application.
Run this script, then open the generated 'test_audio' folder in the Audio Analyzer.
"""

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import chirp
import os

def generate_test_audio():
    output_dir = "test_audio"
    os.makedirs(output_dir, exist_ok=True)

    sr = 44100  # Standard CD quality sample rate
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    print(f"Generating test audio files in '{output_dir}/'...\n")

    # 1. Stereo Sine Waves (Left: 440Hz A4, Right: 880Hz A5)
    # Good for: Testing channel separation, basic spectrum, and level meters
    left = np.sin(2 * np.pi * 440.0 * t)
    right = np.sin(2 * np.pi * 880.0 * t)
    stereo_sine = np.column_stack((left, right))
    wavfile.write(os.path.join(output_dir, "1_stereo_sine_440_880Hz.wav"), sr, stereo_sine)
    print("✔️ 1_stereo_sine_440_880Hz.wav (Stereo, Dual Tone)")

    # 2. Frequency Sweep (Chirp from 100Hz to 5000Hz)
    # Good for: Testing Spectrogram visualization and frequency axis
    sweep = chirp(t, f0=100, f1=5000, t1=duration, method='linear')
    wavfile.write(os.path.join(output_dir, "2_linear_chirp_sweep.wav"), sr, sweep)
    print("✔️ 2_linear_chirp_sweep.wav (Mono, 100Hz->5kHz Sweep)")

    # 3. White Noise mixed with a 1kHz Tone
    # Good for: Testing noise floor, spectral centroid, and dynamic range
    noise = np.random.uniform(-0.6, 0.6, len(t))
    tone_1k = np.sin(2 * np.pi * 1000.0 * t) * 0.4
    mix = noise + tone_1k
    mix = mix / np.max(np.abs(mix))  # Normalize to prevent clipping
    wavfile.write(os.path.join(output_dir, "3_noise_with_1kHz_tone.wav"), sr, mix)
    print("✔️ 3_noise_with_1kHz_tone.wav (Mono, Noisy Signal)")

    # 4. High Dynamic Range (Loud and Quiet sections)
    # Good for: Testing RMS history, Crest Factor, and Dynamic Range metrics
    loud = np.sin(2 * np.pi * 250.0 * t)
    # Create an amplitude envelope that oscillates between 1.0 and 0.01 (-40dB)
    envelope = 0.505 + 0.495 * np.sin(2 * np.pi * 0.5 * t) # 0.5 Hz LFO
    dynamic = loud * envelope
    wavfile.write(os.path.join(output_dir, "4_dynamic_range_test.wav"), sr, dynamic)
    print("✔️ 4_dynamic_range_test.wav (Mono, Pulsing Volume)")

    # 5. Rich Harmonics (Sawtooth-like tone at 110Hz)
    # Good for: Testing Spectrum peaks and Band Energy distribution
    harmonics = np.zeros_like(t)
    for k in range(1, 15): # 14 harmonics
        harmonics += (1.0 / k) * np.sin(2 * np.pi * 110.0 * k * t)
    harmonics = harmonics / np.max(np.abs(harmonics))
    wavfile.write(os.path.join(output_dir, "5_harmonics_110Hz.wav"), sr, harmonics)
    print("✔️ 5_harmonics_110Hz.wav (Mono, Rich Harmonics)")

    print("\n✅ Generation complete! Open these files in the Audio Analyzer to test features.")

if __name__ == "__main__":
    generate_test_audio()