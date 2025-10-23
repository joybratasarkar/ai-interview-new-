# apps/ai_interview/tasks/audio_improved.py
# Improved audio processing for AI interviewer with accurate 2-second pause detection

from ai_interview.celery_app import celery_app
from ai_interview.config import REDIS_URL

import numpy as np
import torch
import time
import json
import base64
import redis
import asyncio
import logging
from collections import deque
from datetime import datetime
import redis.asyncio as aioredis
from celery import shared_task

# Load Silero VAD model once
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Optimized constants for real-time AI interviewer
SAMPLE_RATE = 16000
CHUNK_SIZE = 320  # 20ms chunks for better responsiveness  
SILENCE_THRESHOLD = 0.45  # Slightly lower for better sensitivity
NOISE_THRESHOLD = -45  # More sensitive noise detection
PAUSE_SECONDS = 2.0  # Exact 2-second pause requirement
MIN_SPEECH_TIME = 0.3  # Minimum speech before considering pause (300ms)
MAX_BUFFER_TIME = 8.0  # Maximum buffer time in seconds
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_TIME)

# Redis sync client
dis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Setup logging with microsecond precision
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)
logger = logging.getLogger(__name__)

# In-memory state per room
stream_states = {}
paused_rooms = {}


class ImprovedStreamState:
    """Improved stream state for accurate 2-second pause detection"""
    
    def __init__(self):
        self.buffer = deque()
        self.sample_timestamps = deque()  # Track precise timing
        self.speech_segments = []  # Track speech with timing
        self.last_speech_end = None
        self.first_speech_start = None
        self.last_vad_result = False
        self.consecutive_silence_chunks = 0
        
    def add_samples_with_timing(self, samples: np.ndarray, arrival_time: float):
        """Add samples with precise timestamp tracking"""
        self.buffer.extend(samples.tolist())
        
        # Calculate precise timestamps for each sample
        sample_duration = 1.0 / SAMPLE_RATE
        for i in range(len(samples)):
            sample_time = arrival_time + (i * sample_duration)
            self.sample_timestamps.append(sample_time)
        
        # Maintain buffer size
        while len(self.buffer) > MAX_BUFFER_SAMPLES:
            self.buffer.popleft()
            if self.sample_timestamps:
                self.sample_timestamps.popleft()
    
    def calculate_rms_dbfs(self, chunk: np.ndarray) -> float:
        """Improved RMS calculation with noise floor handling"""
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if rms <= 1e-10:
            return -100.0
        return 20 * np.log10(rms + 1e-10)
    
    def detect_speech(self, chunk: np.ndarray) -> tuple[bool, float, float]:
        """
        Precise speech detection combining VAD and volume
        Returns: (is_speech, vad_probability, dbfs_level)
        """
        dbfs = self.calculate_rms_dbfs(chunk)
        
        # VAD model inference
        with torch.no_grad():
            vad_prob = model(torch.from_numpy(chunk).float(), SAMPLE_RATE).item()
        
        # Combined speech detection criteria
        volume_sufficient = dbfs > NOISE_THRESHOLD
        vad_confident = vad_prob > SILENCE_THRESHOLD
        is_speech = volume_sufficient and vad_confident
        
        return is_speech, vad_prob, dbfs
    
    def update_speech_timeline(self, is_speech: bool, chunk_time: float):
        """Update speech timeline with precise timing"""
        chunk_duration = CHUNK_SIZE / SAMPLE_RATE
        
        if is_speech:
            self.consecutive_silence_chunks = 0
            
            if not self.last_vad_result:  # Speech just started
                if self.first_speech_start is None:
                    self.first_speech_start = chunk_time
                
                # Begin new speech segment
                self.speech_segments.append({
                    'start': chunk_time,
                    'end': chunk_time + chunk_duration
                })
            else:
                # Continue current speech segment
                if self.speech_segments:
                    self.speech_segments[-1]['end'] = chunk_time + chunk_duration
        else:
            self.consecutive_silence_chunks += 1
            
            if self.last_vad_result:  # Speech just ended
                self.last_speech_end = chunk_time
        
        self.last_vad_result = is_speech
    
    def has_sufficient_speech(self) -> bool:
        """Check if minimum speech requirement is met"""
        if not self.speech_segments:
            return False
        
        total_speech_duration = sum(
            seg['end'] - seg['start'] for seg in self.speech_segments
        )
        
        return total_speech_duration >= MIN_SPEECH_TIME
    
    def check_for_pause(self, current_time: float) -> tuple[bool, float]:
        """
        Check for accurate 2-second pause after sufficient speech
        Returns: (pause_detected, silence_duration)
        """
        if not self.has_sufficient_speech():
            return False, 0.0
        
        if self.last_speech_end is None:
            return False, 0.0
        
        # Calculate exact silence duration
        silence_duration = current_time - self.last_speech_end
        
        # Pause detected if silence >= 2.0 seconds
        pause_detected = silence_duration >= PAUSE_SECONDS
        
        return pause_detected, silence_duration
    
    def process_audio_chunk(self) -> tuple[bool, dict]:
        """
        Process one audio chunk with detailed analysis
        Returns: (pause_detected, analysis_metrics)
        """
        if len(self.buffer) < CHUNK_SIZE:
            return False, {}
        
        # Extract chunk with timing
        chunk = np.array([self.buffer.popleft() for _ in range(CHUNK_SIZE)], dtype=np.float32)
        chunk_time = self.sample_timestamps.popleft() if self.sample_timestamps else time.time()
        
        # Analyze chunk
        is_speech, vad_prob, dbfs = self.detect_speech(chunk)
        
        # Update timeline
        self.update_speech_timeline(is_speech, chunk_time)
        
        # Check pause condition
        current_time = time.time()
        pause_detected, silence_duration = self.check_for_pause(current_time)
        
        # Return analysis metrics
        metrics = {
            'is_speech': is_speech,
            'vad_probability': round(vad_prob, 3),
            'dbfs_level': round(dbfs, 2),
            'silence_duration': round(silence_duration, 3),
            'speech_segments': len(self.speech_segments),
            'sufficient_speech': self.has_sufficient_speech(),
            'chunk_timestamp': chunk_time,
            'silence_chunks': self.consecutive_silence_chunks
        }
        
        return pause_detected, metrics


@celery_app.task(name="ai_interview.tasks.audio_improved.process_audio")
def process_audio(payload):
    """Improved audio processing with accurate 2-second pause detection"""
    process_start = time.time()
    
    # Parse input
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            logger.error("Invalid JSON payload received")
            return None
    
    room_id = payload.get("room_id", "default")
    audio_b64 = payload.get("text")
    
    if not audio_b64:
        logger.warning(f"No audio data received for room {room_id}")
        return None
    
    # Check if turn already ended
    if dis.get(f"turn-ended:{room_id}") == "1":
        logger.info(f"Turn already ended for room {room_id}, skipping")
        return None
    
    try:
        # Decode audio blob
        raw_audio = base64.b64decode(audio_b64)
        samples = np.frombuffer(raw_audio, dtype=np.float32)
        
        if len(samples) == 0:
            logger.warning(f"Empty audio samples for room {room_id}")
            return None
            
    except Exception as e:
        logger.error(f"Audio decoding failed for room {room_id}: {e}")
        return None
    
    # Get or create stream state
    if room_id not in stream_states:
        stream_states[room_id] = ImprovedStreamState()
    
    state = stream_states[room_id]
    
    # Add samples with precise timing
    arrival_time = time.time()
    state.add_samples_with_timing(samples, arrival_time)
    
    # Process all available chunks
    pause_detected = False
    final_metrics = None
    
    while len(state.buffer) >= CHUNK_SIZE:
        chunk_pause_detected, metrics = state.process_audio_chunk()
        final_metrics = metrics
        
        if chunk_pause_detected:
            pause_detected = True
            logger.info(f"✅ Accurate 2-second pause detected in room '{room_id}' - "
                       f"Silence: {metrics['silence_duration']:.3f}s")
            break
    
    # Log processing performance
    processing_time = (time.time() - process_start) * 1000
    if final_metrics and processing_time > 50:  # Log if processing takes > 50ms
        logger.warning(f"Slow audio processing for {room_id}: {processing_time:.1f}ms")
    
    # Return result if pause detected
    if pause_detected:
        return _submit_pause_result(room_id, final_metrics)
    
    return None


def _submit_pause_result(room_id: str, metrics: dict) -> dict:
    """Submit pause detection result with cleanup"""
    logger.info(f"Submitting pause result for room '{room_id}' - "
               f"Silence: {metrics['silence_duration']:.3f}s")
    
    # Clear state
    stream_states.pop(room_id, None)
    paused_rooms[room_id] = datetime.now()
    
    # Mark turn ended in Redis
    dis.setex(f"turn-ended:{room_id}", 300, "1")
    
    # Schedule async cleanup
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_cleanup_room(room_id))
    except RuntimeError:
        asyncio.run(_cleanup_room(room_id))
    
    # Return detailed result
    return {
        "room_id": room_id,
        "task_type": "audio",
        "result": True,
        "timestamp": datetime.now().isoformat(),
        "event": "pause_detected",
        "pause_duration_seconds": metrics['silence_duration'],
        "speech_segments_detected": metrics['speech_segments'],
        "final_dbfs_level": metrics['dbfs_level'],
        "final_vad_confidence": metrics['vad_probability'],
        "processing_mode": "real_time_optimized"
    }


async def _cleanup_room(room_id: str):
    """Clean up Redis keys for room"""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.delete(f"stream:{room_id}", f"turn-ended:{room_id}")
    await r.close()
    paused_rooms.pop(room_id, None)


@shared_task(name="ai_interview.tasks.audio_improved.cleanup_audio")
def cleanup_audio(room_id: str):
    """Audio cleanup task"""
    try:
        if room_id in stream_states:
            # Force cleanup with artificial metrics
            forced_metrics = {
                "silence_duration": 2.0,
                "speech_segments": 0,
                "dbfs_level": -100,
                "vad_probability": 0.0
            }
            result = _submit_pause_result(room_id, forced_metrics)
            logger.info(f"Forced audio cleanup for {room_id}: {result}")
            return result
    except Exception as e:
        logger.error(f"Cleanup failed for {room_id}: {e}")


# Utility function for monitoring audio state
def get_audio_room_status(room_id: str) -> dict:
    """Get current audio processing status for monitoring"""
    if room_id not in stream_states:
        return {"error": "Room not found", "room_id": room_id}
    
    state = stream_states[room_id]
    return {
        "room_id": room_id,
        "buffer_samples": len(state.buffer),
        "buffer_duration_ms": (len(state.buffer) / SAMPLE_RATE) * 1000,
        "speech_segments": len(state.speech_segments),
        "has_sufficient_speech": state.has_sufficient_speech(),
        "currently_speaking": state.last_vad_result,
        "silence_chunks": state.consecutive_silence_chunks,
        "last_speech_end": state.last_speech_end
    }