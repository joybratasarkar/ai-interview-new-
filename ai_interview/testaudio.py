# test_pause_accuracy.py

import numpy as np
import torch
import json
import base64
from collections import deque

# Import your StreamState and process_chunk logic
from ai_interview.tasks.audio import StreamState

# We'll monkey‑patch time.time inside that module
import time as real_time
import importlib
import ai_interview.tasks.audio as audio_mod

def test_pause():
    # 1) Monkey‑patch time.time in the audio module
    fake_time = [0.0]
    def fake_time_func():
        return fake_time[0]
    audio_mod.time.time = fake_time_func

    # 2) Create a fresh StreamState
    state = StreamState()

    # 3) Prepare chunks
    CHUNK_SIZE = 512
    # “Speech” chunk: random noise
    speech_chunk = (np.random.randn(CHUNK_SIZE) * 0.1).astype(np.float32)
    # “Silence” chunk: zeros
    silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

    # 4) Simulate 1 second of speech
    chunk_dt = CHUNK_SIZE / float(audio_mod.SAMPLE_RATE)  # ≈0.032s
    num_speech = int(1.0 / chunk_dt)
    for _ in range(num_speech):
        state.process_chunk(speech_chunk)
        fake_time[0] += chunk_dt

    print(f"Fed {num_speech} speech chunks ({num_speech*chunk_dt:.2f}s)")

    # 5) Now feed silence until pause is detected
    max_chunks = int((audio_mod.PAUSE_SECONDS + 3.0) / chunk_dt)
    for i in range(max_chunks):
        detected = state.process_chunk(silent_chunk)
        fake_time[0] += chunk_dt
        if detected:
            print(f"✅ Pause detected after {(i+1)*chunk_dt:.3f}s of silence ({i+1} chunks).")
            break
    else:
        total = max_chunks * chunk_dt
        print(f"❌ No pause detected in {total:.2f}s (expected ~{audio_mod.PAUSE_SECONDS}s).")

if __name__ == "__main__":
    test_pause()
