from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import base64
import numpy as np
import time
import torch
from scipy.io import wavfile
from apps.ai_interview.smart_turn.inference import predict_endpoint

router = APIRouter()

# --- Config ---
RATE = 16000
CHUNK = 512
STOP_MS = 1000
PRE_SPEECH_MS = 200
MAX_DURATION_SECONDS = 16
VAD_THRESHOLD = 0.7
TEMP_OUTPUT_WAV = "temp_ws_output.wav"

# Load Silero VAD model
torch.hub.set_dir("./.torch_hub")
MODEL, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    source="github",
    model="silero_vad",
    onnx=False,
    trust_repo=True,
)

# Per-client state
STATE = {}

def get_state(client_id):
    return STATE.get(client_id, {
        "buffer": [],
        "silence_frames": 0,
        "speech_triggered": False,
        "speech_start": None
    })

def save_state(client_id, st):
    STATE[client_id] = st

def clear_state(client_id):
    if client_id in STATE:
        del STATE[client_id]

def process_segment(audio_buffer, speech_start_time):
    """Process a full speech segment once silence is detected."""
    start_time = speech_start_time - (PRE_SPEECH_MS / 1000)
    start_index = 0
    for i, (t, _) in enumerate(audio_buffer):
        if t >= start_time:
            start_index = i
            break

    segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index:]]
    segment_audio = np.concatenate(segment_audio_chunks)

    # Remove trailing silence more conservatively (only 200ms)
    samples_to_remove = int(200 / 1000 * RATE)  # Only remove 200ms
    if len(segment_audio) > samples_to_remove:
        segment_audio = segment_audio[:-samples_to_remove]

    # limit max duration
    if len(segment_audio) / RATE > MAX_DURATION_SECONDS:
        segment_audio = segment_audio[: int(MAX_DURATION_SECONDS * RATE)]

    # save for debugging
    wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio * 32767).astype(np.int16))

    # predict
    result = predict_endpoint(segment_audio)
    label = "Complete" if result["prediction"] == 1 else "Incomplete"
    return {
        "prediction": label,
        "probability": result["probability"]
    }

def decode_audio_base64(b64_str, target_rate=16000):
    """Decode base64 PCM16 WAV chunk."""
    pcm = base64.b64decode(b64_str.split(",", 1)[-1])
    arr16 = np.frombuffer(pcm, dtype=np.int16)
    return arr16.astype(np.float32) / 32767.0

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"[WS] Connected: {client_id}")

    state = get_state(client_id)

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)

            if payload.get("type") != "audio":
                await websocket.send_json({"error": "Only 'audio' type supported"})
                continue

            b64 = payload.get("audio") or payload.get("text")
            if not b64:
                await websocket.send_json({"error": "Missing audio field"})
                continue

            # decode PCM16 -> float32
            audio_float32 = decode_audio_base64(b64, RATE)
            now = time.time()

            # --- Split audio into 512-sample frames for VAD ---
            frame_size = 512
            speech_detected = False
            speech_prob = 0.0
            for start in range(0, len(audio_float32), frame_size):
                chunk = audio_float32[start:start+frame_size]
                if len(chunk) < frame_size:
                    continue  # skip incomplete frame
                prob = MODEL(torch.from_numpy(chunk).unsqueeze(0), RATE).item()
                speech_prob = prob
                if prob > VAD_THRESHOLD:
                    speech_detected = True
                    break
            is_speech = speech_detected

            # --- Logic copied from record_and_predict.py ---
            if is_speech:
                if not state["speech_triggered"]:
                    state["silence_frames"] = 0
                state["speech_triggered"] = True
                if state["speech_start"] is None:
                    state["speech_start"] = now
                state["buffer"].append((now, audio_float32))
                save_state(client_id, state)

            else:
                if state["speech_triggered"]:
                    state["buffer"].append((now, audio_float32))
                    state["silence_frames"] += 1
                    if state["silence_frames"] * (CHUNK / RATE) >= STOP_MS / 1000:
                        # finalize segment
                        result = process_segment(state["buffer"], state["speech_start"])
                        clear_state(client_id)
                        await websocket.send_json({"status": "segment", **result})
                        state = get_state(client_id)  # reset
                    else:
                        save_state(client_id, state)
                else:
                    # keep short history of silence for pre-speech context
                    state["buffer"].append((now, audio_float32))
                    max_buffer_time = (PRE_SPEECH_MS + STOP_MS) / 1000.0 + MAX_DURATION_SECONDS
                    state["buffer"] = [(t, c) for t, c in state["buffer"] if t >= now - max_buffer_time]
                    save_state(client_id, state)

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {client_id}")
        clear_state(client_id)
