# Test script to verify pause detection sends results

import base64
import numpy as np
import json
import time
from ai_interview.tasks.audio import process_audio

def test_pause_detection_result():
    """Test that pause detection returns proper result"""
    
    print("🧪 Testing pause detection result sending...")
    
    sample_rate = 16000
    room_id = "test_pause_result"
    
    # Step 1: Send speech data for 1 second
    print("\n📢 Step 1: Sending 1 second of speech...")
    speech_duration = 1.0
    speech_samples = np.random.normal(0, 0.2, int(sample_rate * speech_duration)).astype(np.float32)
    audio_b64 = base64.b64encode(speech_samples.tobytes()).decode('utf-8')
    
    payload = {
        "room_id": room_id,
        "text": audio_b64,
        "type": "audio"
    }
    
    result = process_audio(payload)
    print(f"Speech result: {result}")
    
    # Step 2: Send silence for 2.5 seconds
    print("\n🔇 Step 2: Sending 2.5 seconds of silence...")
    silence_duration = 2.5
    silence_samples = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
    audio_b64 = base64.b64encode(silence_samples.tobytes()).decode('utf-8')
    
    payload = {
        "room_id": room_id,
        "text": audio_b64,
        "type": "audio"
    }
    
    result = process_audio(payload)
    print(f"Silence result: {result}")
    
    if result:
        print("\n✅ SUCCESS: Pause detection returned result!")
        print(f"📋 Result details:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    else:
        print("\n❌ FAILURE: No result returned from pause detection")
    
    return result

def test_realtime_chunks():
    """Test with realistic real-time audio chunks"""
    
    print("\n🎯 Testing with real-time audio chunks...")
    
    sample_rate = 16000
    chunk_duration = 0.1  # 100ms chunks (realistic for WebRTC)
    chunk_size = int(sample_rate * chunk_duration)
    room_id = "test_realtime"
    
    # Send 5 speech chunks (500ms total)
    print("📢 Sending speech chunks...")
    for i in range(5):
        speech_chunk = np.random.normal(0, 0.15, chunk_size).astype(np.float32)
        audio_b64 = base64.b64encode(speech_chunk.tobytes()).decode('utf-8')
        
        payload = {
            "room_id": room_id,
            "text": audio_b64,
            "type": "audio"
        }
        
        result = process_audio(payload)
        print(f"  Chunk {i+1}: result={result}")
        if result:
            print(f"⚠️ Unexpected result at speech chunk {i+1}")
            return result
        
        time.sleep(0.05)  # Small delay to simulate real-time
    
    # Send silence chunks until pause is detected
    print("🔇 Sending silence chunks...")
    for i in range(30):  # Up to 3 seconds of silence
        silence_chunk = np.zeros(chunk_size, dtype=np.float32)
        audio_b64 = base64.b64encode(silence_chunk.tobytes()).decode('utf-8')
        
        payload = {
            "room_id": room_id,
            "text": audio_b64,
            "type": "audio"
        }
        
        result = process_audio(payload)
        print(f"  Silence chunk {i+1} ({(i+1)*100}ms): result={result}")
        
        if result:
            print(f"\n✅ Pause detected after {(i+1)*100}ms of silence!")
            print(f"📋 Result: {result}")
            return result
        
        time.sleep(0.05)
    
    print("\n❌ No pause detected after 3 seconds of silence")
    return None

if __name__ == "__main__":
    print("🚀 Testing pause detection result sending")
    print("=" * 60)
    
    # Test 1: Large chunks
    result1 = test_pause_detection_result()
    
    print("\n" + "=" * 60)
    
    # Test 2: Real-time chunks  
    result2 = test_realtime_chunks()
    
    print("\n" + "=" * 60)
    print("🏁 Test Summary:")
    print(f"Large chunk test: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"Real-time test: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 or result2:
        print("\n🎯 Pause detection is working and sending results!")
    else:
        print("\n⚠️ Pause detection may need debugging")