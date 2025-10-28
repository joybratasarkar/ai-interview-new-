import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class LLMPerformanceLogger:
    """Comprehensive logging system for LLM call performance analysis"""
    
    def __init__(self, log_dir="logs"):
        self.session_logs = []
        self.current_request_logs = []
        self.request_start_time = None
        self.total_llm_calls = 0
        self.total_llm_time = 0.0
        
        # File output configuration
        self.log_dir = log_dir
        self.enable_file_logging = True
        
        # Create logs directory if it doesn't exist
        if self.enable_file_logging:
            import os
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"📁 Log files will be saved to: {os.path.abspath(self.log_dir)}")
        
    def start_request(self, request_id: str = None):
        """Start tracking a new request"""
        self.request_start_time = time.time()
        self.current_request_logs = []
        self.request_id = request_id or f"req_{int(time.time())}"
        print(f"\n🚀 === LLM PERFORMANCE TRACKING STARTED: {self.request_id} ===")
        print(f"📅 Timestamp: {datetime.now().isoformat()}")
        
    def log_llm_call(self, 
                     call_name: str, 
                     start_time: float, 
                     end_time: float, 
                     model: str = "gemini-2.0-flash-lite-001",
                     prompt_length: int = 0,
                     response_length: int = 0,
                     success: bool = True,
                     error: str = None,
                     context: str = ""):
        """Log individual LLM call with detailed metrics"""
        duration = end_time - start_time
        self.total_llm_calls += 1
        self.total_llm_time += duration
        
        log_entry = {
            "call_sequence": self.total_llm_calls,
            "call_name": call_name,
            "model": model,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": round(duration, 3),
            "prompt_length": prompt_length,
            "response_length": response_length,
            "success": success,
            "error": error,
            "context": context,
            "timestamp": datetime.fromtimestamp(start_time).isoformat()
        }
        
        self.current_request_logs.append(log_entry)
        
        # Real-time console logging
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n🧠 LLM CALL #{self.total_llm_calls}: {call_name}")
        print(f"   📊 Model: {model}")
        print(f"   ⏱️  Duration: {duration:.3f}s")
        print(f"   📤 Prompt Length: {prompt_length} chars")
        print(f"   📥 Response Length: {response_length} chars")
        print(f"   {status}")
        if context:
            print(f"   📝 Context: {context}")
        if error:
            print(f"   🚨 Error: {error}")
            
    def end_request(self):
        """End request tracking and generate summary"""
        if not self.request_start_time:
            return
            
        total_request_time = time.time() - self.request_start_time
        
        # Calculate actual LLM time considering parallel execution
        if len(self.current_request_logs) > 1:
            # For parallel calls, use max end time - min start time instead of sum
            start_times = [call['start_time'] for call in self.current_request_logs]
            end_times = [call['end_time'] for call in self.current_request_logs]
            actual_llm_time = max(end_times) - min(start_times)
        else:
            actual_llm_time = self.total_llm_time
            
        llm_percentage = (actual_llm_time / total_request_time * 100) if total_request_time > 0 else 0
        non_llm_time = max(0, total_request_time - actual_llm_time)  # Prevent negative values
        
        # Generate summary
        summary = {
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat(),
            "total_request_time": round(total_request_time, 3),
            "total_llm_calls": len(self.current_request_logs),
            "total_llm_time": round(self.total_llm_time, 3),  # Keep sum for individual call analysis
            "actual_llm_time": round(actual_llm_time, 3),  # Parallel-aware timing
            "llm_time_percentage": round(llm_percentage, 1),
            "non_llm_time": round(non_llm_time, 3),
            "average_llm_time": round(self.total_llm_time / len(self.current_request_logs), 3) if self.current_request_logs else 0,
            "llm_calls": self.current_request_logs
        }
        
        self.session_logs.append(summary)
        
        # SAVE TO FILES
        if self.enable_file_logging:
            self._save_request_log(summary)
            self._save_daily_summary()
        
        # Console summary
        print(f"\n📊 === REQUEST PERFORMANCE SUMMARY: {self.request_id} ===")
        print(f"⏱️  Total Request Time: {total_request_time:.3f}s")
        print(f"🧠 Total LLM Calls: {len(self.current_request_logs)}")
        print(f"⚡ Actual LLM Time: {actual_llm_time:.3f}s ({llm_percentage:.1f}%)")
        print(f"📊 Sum LLM Time: {self.total_llm_time:.3f}s (individual call sum)")
        print(f"🔧 Non-LLM Time: {non_llm_time:.3f}s ({100-llm_percentage:.1f}%)")
        print(f"📈 Average LLM Time: {summary['average_llm_time']:.3f}s")
        
        # Performance analysis
        if total_request_time > 5:
            print(f"⚠️  PERFORMANCE WARNING: Request took {total_request_time:.3f}s (>5s)")
        if len(self.current_request_logs) > 4:
            print(f"⚠️  LLM USAGE WARNING: {len(self.current_request_logs)} LLM calls (>4 calls)")
            
        print(f"🏁 === END PERFORMANCE TRACKING ===\n")
        
        # Reset for next request
        self.request_start_time = None
        self.current_request_logs = []
        self.total_llm_calls = 0
        self.total_llm_time = 0.0
        
        return summary
        
    def get_session_stats(self):
        """Get overall session statistics"""
        if not self.session_logs:
            return None
            
        total_requests = len(self.session_logs)
        total_time = sum(log['total_request_time'] for log in self.session_logs)
        total_llm_calls = sum(log['total_llm_calls'] for log in self.session_logs)
        total_llm_time = sum(log['total_llm_time'] for log in self.session_logs)
        
        return {
            "total_requests": total_requests,
            "total_session_time": round(total_time, 3),
            "total_llm_calls": total_llm_calls,
            "total_llm_time": round(total_llm_time, 3),
            "average_request_time": round(total_time / total_requests, 3),
            "average_llm_calls_per_request": round(total_llm_calls / total_requests, 1),
            "requests": self.session_logs
        }
    
    def _save_request_log(self, summary):
        """Save individual request log to file"""
        import os
        today = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(self.log_dir, f"interview_requests_{today}.jsonl")
        
        try:
            with open(filename, 'a') as f:
                f.write(json.dumps(summary) + '\n')
            print(f"💾 Request log saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save request log: {e}")
    
    def _save_daily_summary(self):
        """Save daily summary to file"""
        import os
        today = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(self.log_dir, f"daily_summary_{today}.json")
        
        if not self.session_logs:
            return
            
        # Calculate daily stats
        total_requests = len(self.session_logs)
        total_time = sum(log['total_request_time'] for log in self.session_logs)
        total_llm_calls = sum(log['total_llm_calls'] for log in self.session_logs)
        total_llm_time = sum(log['total_llm_time'] for log in self.session_logs)
        
        daily_summary = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "total_requests": total_requests,
            "total_session_time": round(total_time, 3),
            "total_llm_calls": total_llm_calls,
            "total_llm_time": round(total_llm_time, 3),
            "average_request_time": round(total_time / total_requests, 3) if total_requests > 0 else 0,
            "average_llm_calls_per_request": round(total_llm_calls / total_requests, 1) if total_requests > 0 else 0,
            "requests": self.session_logs
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(daily_summary, f, indent=2)
            print(f"📊 Daily summary updated: {filename}")
        except Exception as e:
            print(f"❌ Failed to save daily summary: {e}")

# Global logger instance
llm_performance_logger = LLMPerformanceLogger()