import time
import numpy as np
import torch
from ai_interview.smart_turn.inference import predict_endpoint


class OptimizedSmartTurnDetector:
    """
    Optimized turn detection system that combines Silero VAD with smart-turn ML model
    for adaptive timeout based on speech completion confidence.
    
    Key improvements:
    - Limits audio processing to prevent slowdown on long speech
    - Uses adaptive timeouts based on ML confidence
    - Prevents predictions during active speech
    - Caches predictions to avoid redundant processing
    """
    
    def __init__(self):
        # Smart-turn ML model
        self.smart_turn_model = predict_endpoint
        
        # Performance optimization parameters
        self.max_audio_seconds = 12      # Limit audio length for consistent performance
        self.prediction_cooldown = 1.5   # Seconds between ML predictions
        self.energy_threshold = 0.015    # Speech energy threshold
        self.min_silence_for_prediction = 0.4  # Minimum silence before prediction
        
        # Adaptive timeout parameters
        self.base_timeout = 1200         # ms - base timeout
        self.min_timeout = 400           # ms - for high confidence predictions
        self.max_timeout = 2500          # ms - for low confidence predictions
        self.initial_check_delay = 300   # ms - delay before first prediction
        
        # State tracking
        self.last_prediction_time = 0
        self.last_prediction_result = None
        self.adaptive_timeout = self.base_timeout
        self.silence_start_time = None
        
    def process_audio_chunk(self, is_speech, audio_buffer, conversation_context=""):
        """
        Main processing function called for each audio chunk
        STRICT RULE: Only processes during silence, never during speech
        
        Args:
            is_speech: Boolean from VAD detection
            audio_buffer: List of (timestamp, audio_chunk) tuples
            conversation_context: String context for ML model
            
        Returns:
            str: "CONTINUE_LISTENING" or "END_TURN"
        """
        current_time = time.time()
        
        # STRICT RULE: If speech detected, immediately return without processing
        if is_speech:
            print("Speech detected - resetting silence state, no ML processing")
            self._reset_silence_state()
            return "CONTINUE_LISTENING"
        
        # ONLY PROCESS DURING SILENCE
        silence_duration_ms = self._update_silence_duration(current_time)
        print(f"Silence duration: {silence_duration_ms:.0f}ms")
        
        # Initial delay before any processing
        if silence_duration_ms < self.initial_check_delay:
            return "CONTINUE_LISTENING"
        
        # Run smart-turn prediction if conditions are met (ONLY ONCE)
        if self._should_run_prediction(audio_buffer, silence_duration_ms, current_time):
            print("Running ML prediction...")
            self._run_smart_turn_prediction(audio_buffer, conversation_context)
        
        # Check if adaptive timeout reached
        if silence_duration_ms >= self.adaptive_timeout:
            print(f"Timeout reached: {silence_duration_ms:.0f}ms >= {self.adaptive_timeout}ms")
            return "END_TURN"
        
        return "CONTINUE_LISTENING"
    
    def _reset_silence_state(self):
        """Reset all silence-related state when speech is detected"""
        print("Resetting silence state - clearing predictions")
        self.silence_start_time = None
        self.last_prediction_result = None
        self.adaptive_timeout = self.base_timeout
        self.last_prediction_time = 0  # Allow new predictions
    
    def _update_silence_duration(self, current_time):
        """Update and return current silence duration in milliseconds"""
        if self.silence_start_time is None:
            self.silence_start_time = current_time
            
        silence_duration_ms = (current_time - self.silence_start_time) * 1000
        return silence_duration_ms
    
    def _should_run_prediction(self, audio_buffer, silence_duration_ms, current_time):
        """Determine if we should run the expensive ML prediction - STRICT CONTROL"""
        
        # RULE 1: Only predict ONCE per silence period
        if self.last_prediction_result is not None:
            print("Prediction already made for this silence period")
            return False
        
        # RULE 2: Don't predict too frequently (performance)
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            print(f"Cooldown active: {current_time - self.last_prediction_time:.1f}s < {self.prediction_cooldown}s")
            return False
        
        # RULE 3: Minimum silence required
        if silence_duration_ms < (self.min_silence_for_prediction * 1000):
            print(f"Insufficient silence: {silence_duration_ms:.0f}ms < {self.min_silence_for_prediction * 1000}ms")
            return False
        
        # RULE 4: Check for ongoing speech (energy threshold)
        recent_energy = self._calculate_recent_energy(audio_buffer)
        if recent_energy > self.energy_threshold:
            print(f"Speech energy too high: {recent_energy:.4f} > {self.energy_threshold}")
            return False
        
        print("All conditions met - running ML prediction")
        return True
    
    def _calculate_recent_energy(self, audio_buffer):
        """Calculate RMS energy of recent audio to detect ongoing speech"""
        if not audio_buffer:
            return 0.0
        
        # Check energy of last 1 second of audio
        cutoff_time = time.time() - 1.0
        recent_chunks = [
            chunk for timestamp, chunk in audio_buffer 
            if timestamp >= cutoff_time
        ]
        
        if not recent_chunks:
            return 0.0
        
        # Calculate RMS energy
        try:
            recent_audio = np.concatenate(recent_chunks)
            rms_energy = np.sqrt(np.mean(recent_audio ** 2))
            return rms_energy
        except Exception as e:
            print(f"Energy calculation error: {e}")
            return 0.0
    
    def _run_smart_turn_prediction(self, audio_buffer, conversation_context):
        """Run smart-turn ML prediction with optimizations"""
        try:
            start_time = time.time()
            
            # Prepare optimized audio segment
            optimized_audio = self._prepare_audio_segment(audio_buffer)
            
            if len(optimized_audio) == 0:
                print("No audio data for prediction")
                return
            
            # Run smart-turn prediction
            result = self.smart_turn_model(optimized_audio)
            prediction_time = (time.time() - start_time) * 1000
            
            print(f"Smart-turn prediction: {result.get('prediction', 'Unknown')} "
                  f"(confidence: {result.get('probability', 0.0):.3f}) "
                  f"in {prediction_time:.1f}ms")
            
            # Calculate adaptive timeout based on confidence
            completion_confidence = result.get('probability', 0.5)
            self.adaptive_timeout = self._calculate_adaptive_timeout(completion_confidence)
            
            # Cache result and timing
            self.last_prediction_result = result
            self.last_prediction_time = time.time()
            
            print(f"Adaptive timeout set to: {self.adaptive_timeout}ms")
            
        except Exception as e:
            print(f"Smart-turn prediction failed: {e}")
            # Use safe defaults on error
            self.adaptive_timeout = self.base_timeout
    
    def _prepare_audio_segment(self, audio_buffer):
        """Prepare optimized audio segment for ML prediction"""
        if not audio_buffer:
            return np.array([])
        
        # Take only recent audio for consistent performance
        cutoff_time = time.time() - self.max_audio_seconds
        recent_buffer = [
            (timestamp, chunk) for timestamp, chunk in audio_buffer 
            if timestamp >= cutoff_time
        ]
        
        if not recent_buffer:
            return np.array([])
        
        try:
            # Concatenate recent chunks
            audio_chunks = [chunk for _, chunk in recent_buffer]
            segment_audio = np.concatenate(audio_chunks)
            
            # Additional safety: limit to max samples
            max_samples = self.max_audio_seconds * 16000  # Assuming 16kHz
            if len(segment_audio) > max_samples:
                segment_audio = segment_audio[-max_samples:]  # Take most recent
            
            return segment_audio
            
        except Exception as e:
            print(f"Audio segment preparation error: {e}")
            return np.array([])
    
    def _calculate_adaptive_timeout(self, completion_confidence):
        """
        Calculate adaptive timeout based on smart-turn confidence
        Higher confidence = shorter timeout (user likely done)
        Lower confidence = longer timeout (user might continue)
        """
        # Ensure confidence is in valid range
        completion_confidence = max(0.0, min(1.0, completion_confidence))
        
        # Inverse relationship: higher confidence = shorter timeout
        confidence_factor = 1.0 - completion_confidence
        
        # Calculate adaptive timeout using linear interpolation
        timeout_range = self.max_timeout - self.min_timeout
        adaptive_timeout = self.min_timeout + (timeout_range * confidence_factor)
        
        # Ensure bounds
        adaptive_timeout = max(self.min_timeout, min(self.max_timeout, adaptive_timeout))
        
        return int(adaptive_timeout)
    
    def get_current_timeout(self):
        """Get the current adaptive timeout value"""
        return self.adaptive_timeout
    
    def get_last_prediction(self):
        """Get the last ML prediction result"""
        return self.last_prediction_result
    
    def reset_state(self):
        """Reset all state for new conversation"""
        self.last_prediction_time = 0
        self.last_prediction_result = None
        self.adaptive_timeout = self.base_timeout
        self.silence_start_time = None
        print("Smart-turn detector state reset")