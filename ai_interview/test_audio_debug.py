# Test script to debug audio processing issues

import base64
import numpy as np
import json
from ai_interview.tasks.audio import process_audio

def test_audio_processing():
    """Test audio processing with synthetic data"""
    
    # Create synthetic audio data (2 seconds of speech + 2.5 seconds of silence)
    sample_rate = 16000
    
    # Speech segment (random noise to simulate speech)
    speech_duration = 2.0  # 2 seconds
    speech_samples = np.random.normal(0, 0.1, int(sample_rate * speech_duration)).astype(np.float32)
    
    # Silence segment 
    silence_duration = 2.5  # 2.5 seconds (should trigger pause detection)
    silence_samples = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
    
    # Combine speech + silence
    audio_data = np.concatenate([speech_samples, silence_samples])
    
    # Encode as base64
    audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    # Create payload
    payload = {
        "room_id": "test_room_123",
        "text": audio_b64,
        "type": "audio"
    }
    
    print(f"🧪 Testing audio processing with {len(audio_data)} samples")
    print(f"📊 Speech: {speech_duration}s, Silence: {silence_duration}s")
    
    # Process audio
    result = process_audio(payload)
    
    print(f"🎯 Result: {result}")
    
    if result:
        print("✅ SUCCESS: Pause detected correctly!")
    else:
        print("❌ FAILURE: No pause detected")
    
    return result

def test_multiple_chunks():
    """Test with multiple small chunks (simulating real-time blobs)"""
    
    sample_rate = 16000
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(sample_rate * chunk_duration)
    
    room_id = "test_room_chunks"
    
    print(f"🧪 Testing with multiple {chunk_duration*1000}ms chunks")
    
    # Send speech chunks for 1 second
    for i in range(10):  # 10 x 100ms = 1 second of speech
        speech_chunk = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
        audio_b64 = base64.b64encode(speech_chunk.tobytes()).decode('utf-8')
        
        payload = {
            "room_id": room_id,
            "text": audio_b64,
            "type": "audio"
        }
        
        result = process_audio(payload)
        print(f"📦 Chunk {i+1}: result={result}")
        
        if result:
            print(f"⚠️ Unexpected pause detected at chunk {i+1}")
            return result
    
    # Send silence chunks for 2.5 seconds
    for i in range(25):  # 25 x 100ms = 2.5 seconds of silence
        silence_chunk = np.zeros(chunk_size, dtype=np.float32)
        audio_b64 = base64.b64encode(silence_chunk.tobytes()).decode('utf-8')
        
        payload = {
            "room_id": room_id,
            "text": audio_b64,
            "type": "audio"
        }
        
        result = process_audio(payload)
        print(f"🔇 Silence chunk {i+1}: result={result}")
        
        if result:
            print(f"✅ Pause detected at silence chunk {i+1}")
            return result
    
    print("❌ No pause detected after all chunks")
    return None

if __name__ == "__main__":
    print("🚀 Starting audio processing debug tests")
    
    print("\n" + "="*50)
    print("TEST 1: Single large chunk")
    print("="*50)
    test_audio_processing()
    
    print("\n" + "="*50)
    print("TEST 2: Multiple small chunks")
    print("="*50)
    test_multiple_chunks()
    
    print("\n🏁 Debug tests completed")