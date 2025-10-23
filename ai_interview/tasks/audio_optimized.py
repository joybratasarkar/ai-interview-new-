# apps/ai_interview/tasks/audio_optimized.py
# Optimized audio processing for AI interviewer with accurate 2-second pause detection

from apps.ai_interview.celery_app import celery_app
from apps.ai_interview.config import REDIS_URL

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
CHUNK_SIZE = 320  # 20ms chunks (16000 * 0.02) for better responsiveness
SILENCE_THRESHOLD = 0.45  # Slightly lower for better sensitivity
NOISE_THRESHOLD = -45  # More sensitive noise detection
PAUSE_SECONDS = 2.0  # Exact 2-second pause requirement
MIN_SPEECH_TIME = 0.3  # Minimum speech before considering pause (300ms)
MAX_BUFFER_TIME = 8.0  # Maximum buffer time in seconds
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_TIME)

# Timing precision
TIMING_PRECISION = 0.02  # 20ms precision for timing calculations

# Redis sync client
dis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'  # Include microseconds for precision
)
logger = logging.getLogger(__name__)

# In-memory state per room
stream_states = {}
paused_rooms = {}


class OptimizedStreamState:
    """Optimized stream state for accurate pause detection"""
    
    def __init__(self):
        self.buffer = deque()
        self.sample_timestamps = deque()  # Track when each sample arrived
        self.speech_segments = []  # Track speech segments with precise timing
        self.last_speech_end = None
        self.first_speech_start = None
        self.total_samples_processed = 0
        self.last_vad_result = False
        self.consecutive_silence_chunks = 0
        self.consecutive_speech_chunks = 0
        
    def add_samples(self, samples: np.ndarray, arrival_time: float):
        """Add samples with precise timing"""
        self.buffer.extend(samples.tolist())
        
        # Add timestamp for each sample
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
        """Calculate RMS in dBFS with noise floor handling"""
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if rms <= 1e-10:  # Very quiet
            return -100.0
        return 20 * np.log10(rms + 1e-10)  # Add small epsilon to avoid log(0)
    
    def is_speech_detected(self, chunk: np.ndarray) -> tuple[bool, float, float]:
        """
        Improved speech detection with confidence scoring
        Returns: (is_speech, vad_prob, dbfs_level)
        """
        # Calculate audio levels
        dbfs = self.calculate_rms_dbfs(chunk)
        
        # Run VAD model
        with torch.no_grad():
            vad_prob = model(torch.from_numpy(chunk).float(), SAMPLE_RATE).item()
        
        # Multi-criteria speech detection
        volume_ok = dbfs > NOISE_THRESHOLD
        vad_ok = vad_prob > SILENCE_THRESHOLD
        
        # Consider both VAD and volume
        is_speech = volume_ok and vad_ok
        
        return is_speech, vad_prob, dbfs
    
    def update_speech_tracking(self, is_speech: bool, chunk_time: float):
        """Update speech segment tracking with precise timing"""
        chunk_duration = len(self.get_current_chunk()) / SAMPLE_RATE
        
        if is_speech:
            self.consecutive_speech_chunks += 1
            self.consecutive_silence_chunks = 0
            
            if not self.last_vad_result:  # Speech started
                if self.first_speech_start is None:
                    self.first_speech_start = chunk_time
                
                # Start new speech segment
                self.speech_segments.append({
                    'start': chunk_time,
                    'end': chunk_time + chunk_duration
                })
            else:
                # Continue speech segment
                if self.speech_segments:
                    self.speech_segments[-1]['end'] = chunk_time + chunk_duration
        else:
            self.consecutive_silence_chunks += 1
            self.consecutive_speech_chunks = 0
            
            if self.last_vad_result:  # Speech ended
                self.last_speech_end = chunk_time
        
        self.last_vad_result = is_speech
    
    def get_current_chunk(self) -> np.ndarray:
        """Get current chunk for processing"""
        if len(self.buffer) >= CHUNK_SIZE:
            return np.array(list(self.buffer)[:CHUNK_SIZE], dtype=np.float32)
        return np.array([], dtype=np.float32)
    
    def has_sufficient_speech(self) -> bool:
        """Check if we have enough speech before considering pause"""
        if not self.speech_segments or self.first_speech_start is None:
            return False
        
        # Calculate total speech time
        total_speech_time = sum(
            seg['end'] - seg['start'] for seg in self.speech_segments
        )
        
        return total_speech_time >= MIN_SPEECH_TIME
    
    def check_pause_condition(self, current_time: float) -> tuple[bool, float]:
        """
        Check if 2-second pause condition is met
        Returns: (pause_detected, silence_duration)
        """
        if not self.has_sufficient_speech():
            return False, 0.0
        
        if self.last_speech_end is None:
            return False, 0.0
        
        # Calculate precise silence duration
        silence_duration = current_time - self.last_speech_end
        
        # Check if we have exactly 2 seconds of silence
        pause_detected = silence_duration >= PAUSE_SECONDS
        
        return pause_detected, silence_duration
    
    def process_chunk_optimized(self) -> tuple[bool, dict]:
        """
        Optimized chunk processing with detailed metrics
        Returns: (pause_detected, metrics)
        """
        if len(self.buffer) < CHUNK_SIZE:
            return False, {}
        
        # Get chunk and timing
        chunk = np.array([self.buffer.popleft() for _ in range(CHUNK_SIZE)], dtype=np.float32)
        chunk_start_time = self.sample_timestamps.popleft() if self.sample_timestamps else time.time()
        
        # Speech detection
        is_speech, vad_prob, dbfs = self.is_speech_detected(chunk)
        
        # Update tracking
        self.update_speech_tracking(is_speech, chunk_start_time)
        self.total_samples_processed += CHUNK_SIZE
        
        # Check pause condition
        current_time = time.time()
        pause_detected, silence_duration = self.check_pause_condition(current_time)
        
        # Detailed metrics for debugging/monitoring
        metrics = {
            'is_speech': is_speech,
            'vad_probability': round(vad_prob, 3),
            'dbfs_level': round(dbfs, 2),
            'silence_duration': round(silence_duration, 3),
            'total_speech_segments': len(self.speech_segments),
            'has_sufficient_speech': self.has_sufficient_speech(),
            'chunk_time': chunk_start_time,
            'consecutive_silence_chunks': self.consecutive_silence_chunks,
            'consecutive_speech_chunks': self.consecutive_speech_chunks
        }
        
        return pause_detected, metrics


@celery_app.task(name="apps.ai_interview.tasks.audio_optimized.process_audio_optimized")
def process_audio_optimized(payload):
    """Optimized audio processing with accurate 2-second pause detection"""
    start_time = time.time()
    
    # Parse payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            logger.error("Invalid JSON payload")
            return None
    
    room_id = payload.get("room_id", "default")
    audio_b64 = payload.get("text")
    
    if not audio_b64:
        logger.warning(f"No audio data for room {room_id}")
        return None
    
    # Skip if turn already ended
    if dis.get(f"turn-ended:{room_id}") == "1":
        logger.info(f"Turn already ended for room {room_id}, skipping")
        return None
    
    try:
        # Decode audio with better error handling
        raw_audio = base64.b64decode(audio_b64)
        samples = np.frombuffer(raw_audio, dtype=np.float32)
        
        if len(samples) == 0:
            logger.warning(f"Empty audio samples for room {room_id}")
            return None
            
    except Exception as e:
        logger.error(f"Audio decoding error for room {room_id}: {e}")
        return None
    
    # Get or create optimized stream state
    stream_state = stream_states.setdefault(room_id, OptimizedStreamState())
    
    # Add samples with precise timing
    arrival_time = time.time()
    stream_state.add_samples(samples, arrival_time)
    
    # Process all available chunks
    processing_results = []
    pause_detected = False
    
    while len(stream_state.buffer) >= CHUNK_SIZE:
        chunk_pause_detected, metrics = stream_state.process_chunk_optimized()
        processing_results.append(metrics)
        
        if chunk_pause_detected:
            pause_detected = True
            logger.info(f"✅ Accurate 2-second pause detected in room '{room_id}' - Duration: {metrics['silence_duration']}s")
            break
    
    # Return result if pause detected
    if pause_detected:
        return _submit_and_cleanup_optimized(room_id, processing_results[-1])
    
    # Log processing metrics periodically
    total_processing_time = time.time() - start_time
    if len(processing_results) > 0:
        latest_metrics = processing_results[-1]
        if stream_state.total_samples_processed % (SAMPLE_RATE // 2) == 0:  # Every 0.5 seconds
            logger.debug(f"Room {room_id} - Speech: {latest_metrics['is_speech']}, "
                        f"Silence: {latest_metrics['silence_duration']:.2f}s, "
                        f"Processing: {total_processing_time*1000:.1f}ms")
    
    return None


def _submit_and_cleanup_optimized(room_id: str, final_metrics: dict) -> dict:
    """Optimized cleanup with detailed result metrics"""
    logger.info(f"Final pause submitted for room '{room_id}' with {final_metrics['silence_duration']:.3f}s silence")
    
    # Get final state
    stream_state = stream_states.get(room_id)
    
    # Clear state
    stream_states.pop(room_id, None)
    paused_rooms[room_id] = datetime.now()
    
    # Mark turn ended
    dis.setex(f"turn-ended:{room_id}", 300, "1")
    
    # Async cleanup
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_cleanup_room(room_id))
    except RuntimeError:
        asyncio.run(_cleanup_room(room_id))
    
    # Return detailed result
    result = {
        "room_id": room_id,
        "task_type": "audio_optimized",
        "result": True,
        "timestamp": datetime.now().isoformat(),
        "event": "pause_detected",
        "pause_duration": final_metrics['silence_duration'],
        "total_speech_segments": final_metrics.get('total_speech_segments', 0),
        "final_dbfs": final_metrics.get('dbfs_level', 0),
        "final_vad_prob": final_metrics.get('vad_probability', 0),
        "processing_accuracy": "2_second_precise"
    }
    
    return result


async def _cleanup_room(room_id: str):
    """Clean up Redis keys for room"""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.delete(f"stream:{room_id}", f"turn-ended:{room_id}")
    await r.close()
    paused_rooms.pop(room_id, None)


@shared_task(name="apps.ai_interview.tasks.audio_optimized.cleanup_audio_optimized")
def cleanup_audio_optimized(room_id: str):
    """Optimized audio cleanup task"""
    try:
        if room_id in stream_states:
            final_metrics = {"silence_duration": 2.0, "total_speech_segments": 0}
            result = _submit_and_cleanup_optimized(room_id, final_metrics)
            logger.info(f"Audio cleanup completed for {room_id}: {result}")
            return result
    except Exception as e:
        logger.error(f"CLEANUP TASK ERROR - Failed to clean up audio for {room_id}: {e}")


# Utility function for monitoring
def get_room_audio_stats(room_id: str) -> dict:
    """Get current audio processing statistics for a room"""
    if room_id not in stream_states:
        return {"error": "Room not found"}
    
    state = stream_states[room_id]
    return {
        "room_id": room_id,
        "buffer_size": len(state.buffer),
        "samples_processed": state.total_samples_processed,
        "speech_segments": len(state.speech_segments),
        "has_sufficient_speech": state.has_sufficient_speech(),
        "last_speech_detected": state.last_vad_result,
        "buffer_duration_seconds": len(state.buffer) / SAMPLE_RATE
    }