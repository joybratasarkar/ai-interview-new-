import time
import json
import base64
import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
import redis
from scipy.io import wavfile

from apps.ai_interview.celery_app import celery_app
from apps.ai_interview.smart_turn.inference import predict_endpoint

# ─── Configuration ─────────────────────────────────────────────────────────────
RATE, CHUNK = 16000, 512
VAD_THRESHOLD, PRE_SPEECH_MS, STOP_MS, MAX_SEC = 0.7, 200, 1000, 16
TEMP_WAV = "temp_output.wav"

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load Silero VAD ─────────────────────────────────────────────────────────────
torch.hub.set_dir("./.torch_hub")
MODEL, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    source="github",
    model="silero_vad",
    onnx=False,
    trust_repo=True,
)
logger.info("Silero VAD model loaded.")

# ─── Redis for state ────────────────────────────────────────────────────────────
rdb = redis.Redis()

def _key(room: str, name: str) -> str:
    return f"audio:{room}:{name}"

def _load(room: str) -> Tuple[List[Tuple[float, np.ndarray]], int, bool, float]:
    buf = rdb.get(_key(room, "buffer"))
    buffer = [] if buf is None else pickle.loads(buf)
    silence = int(rdb.get(_key(room, "silence")) or 0)
    trig = bool(int(rdb.get(_key(room, "trig")) or 0))
    start_ts = float(rdb.get(_key(room, "start")) or 0.0)
    return buffer, silence, trig, start_ts

def _save(room: str, buffer, silence, trig, start_ts):
    rdb.set(_key(room, "buffer"), pickle.dumps(buffer))
    rdb.set(_key(room, "silence"), silence)
    rdb.set(_key(room, "trig"), int(trig))
    rdb.set(_key(room, "start"), start_ts)

def _clear(room: str):
    for name in ("buffer","silence","trig","start"):
        rdb.delete(_key(room, name))

def _process_segment(buffer: List[Tuple[float, np.ndarray]], start_ts: float) -> dict:
    # include pre-speech
    st = start_ts - PRE_SPEECH_MS/1000.0
    idx0 = next((i for i,(t,_) in enumerate(buffer) if t>=st), 0)
    idx1 = len(buffer) - 1
    seg = np.concatenate([chunk for _,chunk in buffer[idx0:idx1+1]])
    # trim tail silence
    trim = int((STOP_MS - PRE_SPEECH_MS)/1000.0 * RATE)
    if len(seg) > trim:
        seg = seg[:-trim]
    # enforce max duration
    seg = seg[:int(MAX_SEC * RATE)]
    # save WAV for debugging
    wavfile.write(TEMP_WAV, RATE, (seg * 32767).astype(np.int16))
    logger.info(f"Processing speech segment of length {len(seg)/RATE:.2f}s ...")
    # predict
    t0 = time.perf_counter()
    res = predict_endpoint(seg)
    t1 = time.perf_counter()
    label = "Complete" if res.get("prediction")==1 else "Incomplete"
    logger.info("--------")
    logger.info(f"Prediction: {label}")
    logger.info(f"Probability of complete: {res.get('probability'):.4f}")
    logger.info(f"Prediction took {(t1-t0)*1000:.2f}ms")
    logger.info("Listening for speech...")
    logger.info("--------")
    return {"prediction": label, "probability": res.get("probability", 0.0)}

# ─── Celery Task ────────────────────────────────────────────────────────────────
@celery_app.task(name="apps.ai_interview.tasks.audio.process_audio")
def process_audio(payload) -> dict:
    """Accepts one base64-encoded PCM16 chunk per call, buffers and VADs."""
    # parse
    if isinstance(payload, str):
        payload = json.loads(payload)
    room = payload.get("room_id", "default")
    b64  = payload.get("text") or payload.get("audio")
    if not b64:
        logger.info(f"[{room}] no audio")
        return {"error":"no_audio", "room_id": room}

    # decode to float32 normalized
    pcm = base64.b64decode(b64.split(",",1)[-1])
    arr16 = np.frombuffer(pcm, dtype=np.int16)
    audio_f = arr16.astype(np.float32) / 32767.0

    # load previous state
    buf, sil, trig, start_ts = _load(room)

    # VAD inference
    prob = MODEL(torch.from_numpy(audio_f).unsqueeze(0), RATE).item()
    is_speech = prob > VAD_THRESHOLD
    now = time.time()
    logger.debug(f"[{room}] prob={prob:.3f} speech={is_speech} trig={trig} sil={sil} buf={len(buf)}")

    # ─── Speech ongoing ─────────────────────────────────────────────
    if is_speech:
        if not trig:
            sil = 0
            logger.info(f"[{room}] Speech started")
        trig = True
        if start_ts == 0.0:
            start_ts = now
        buf.append((now, audio_f))
        _save(room, buf, sil, trig, start_ts)
        return {"status":"speech", "probability":prob, "room_id":room}

    # ─── Trailing silence (speech ended?) ──────────────────────────
    if trig:
        logger.info(f"[{room}] trailing silence #{sil+1}")
        buf.append((now, audio_f))
        sil += 1
        if sil * (CHUNK/RATE) >= STOP_MS/1000.0:
            logger.info(f"[{room}] segment ends after {sil} frames of silence")
            result = _process_segment(buf, start_ts)
            _clear(room)
            return {"status":"segment", **result, "room_id":room}
        _save(room, buf, sil, trig, start_ts)
        return {"status":"speech",   "probability":prob, "room_id":room}

    # ─── Pre-speech silence ────────────────────────────────────────
    buf.append((now, audio_f))
    max_buf = (PRE_SPEECH_MS + STOP_MS)/1000.0 + MAX_SEC
    buf = [(t,c) for t,c in buf if t >= now - max_buf]
    _save(room, buf, sil, trig, start_ts)
    # return {"status":"listening", "probability":prob, "room_id":room}
