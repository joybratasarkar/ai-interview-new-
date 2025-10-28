import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from functools import wraps
from dataclasses import dataclass, asdict
from contextlib import contextmanager

@dataclass
class LatencyEntry:
    """Individual latency measurement entry"""
    timestamp: str
    function_name: str
    operation_type: str  # 'llm_call', 'function', 'redis_operation', 'cache_lookup'
    duration_ms: float
    start_time: float
    end_time: float
    thread_id: str
    context: str = ""
    input_size: int = 0
    output_size: int = 0
    success: bool = True
    error_message: str = ""
    model_name: str = ""
    
class DetailedLatencyLogger:
    """Comprehensive latency tracking system for interview flow optimization"""
    
    def __init__(self, log_to_file: bool = True):
        self.entries: List[LatencyEntry] = []
        self.request_start_time: Optional[float] = None
        self.request_id: str = ""
        self.log_to_file = log_to_file
        self._lock = threading.Lock()
        
        # Performance categories
        self.llm_calls = []
        self.function_calls = []
        self.cache_operations = []
        self.redis_operations = []
        
    def start_request_tracking(self, request_id: str):
        """Start tracking a new request"""
        with self._lock:
            self.request_start_time = time.time()
            self.request_id = request_id
            self.entries.clear()
            self.llm_calls.clear()
            self.function_calls.clear()
            self.cache_operations.clear()
            self.redis_operations.clear()
            
        print(f"\n🔍 === LATENCY TRACKING STARTED: {request_id} ===")
        print(f"📅 Start Time: {datetime.now().isoformat()}")
        
    def log_operation(self, 
                     function_name: str,
                     operation_type: str,
                     duration_ms: float,
                     start_time: float,
                     end_time: float,
                     context: str = "",
                     input_size: int = 0,
                     output_size: int = 0,
                     success: bool = True,
                     error_message: str = "",
                     model_name: str = ""):
        """Log a single operation with detailed metrics"""
        
        entry = LatencyEntry(
            timestamp=datetime.fromtimestamp(start_time).isoformat(),
            function_name=function_name,
            operation_type=operation_type,
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time,
            thread_id=str(threading.get_ident()),
            context=context,
            input_size=input_size,
            output_size=output_size,
            success=success,
            error_message=error_message,
            model_name=model_name
        )
        
        with self._lock:
            self.entries.append(entry)
            
            # Categorize for analysis
            if operation_type == 'llm_call':
                self.llm_calls.append(entry)
            elif operation_type == 'function':
                self.function_calls.append(entry)
            elif operation_type == 'cache_lookup':
                self.cache_operations.append(entry)
            elif operation_type == 'redis_operation':
                self.redis_operations.append(entry)
        
        # Real-time logging
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} [{operation_type.upper()}] {function_name}: {duration_ms:.2f}ms")
        if context:
            print(f"    📝 Context: {context}")
        if not success and error_message:
            print(f"    🚨 Error: {error_message}")
            
    def function_timer(self, operation_type: str = "function", context: str = ""):
        """Decorator to automatically time function execution"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                function_name = func.__name__
                success = True
                error_message = ""
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    # Calculate input/output sizes if possible
                    input_size = 0
                    output_size = 0
                    
                    if args:
                        try:
                            input_size = len(str(args[0])) if args[0] else 0
                        except:
                            pass
                    
                    if result:
                        try:
                            output_size = len(str(result)) if result else 0
                        except:
                            pass
                    
                    self.log_operation(
                        function_name=function_name,
                        operation_type=operation_type,
                        duration_ms=duration_ms,
                        start_time=start_time,
                        end_time=end_time,
                        context=context,
                        input_size=input_size,
                        output_size=output_size,
                        success=success,
                        error_message=error_message
                    )
                    
            return wrapper
        return decorator
    
    @contextmanager
    def time_operation(self, function_name: str, operation_type: str, context: str = ""):
        """Context manager for timing operations"""
        start_time = time.time()
        success = True
        error_message = ""
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            self.log_operation(
                function_name=function_name,
                operation_type=operation_type,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                context=context,
                success=success,
                error_message=error_message
            )
    
    def log_llm_call(self, model_name: str, function_name: str, prompt_length: int, response_length: int,
                     start_time: float, end_time: float, success: bool = True, error_message: str = "", context: str = ""):
        """Specialized logging for LLM calls"""
        duration_ms = (end_time - start_time) * 1000
        
        self.log_operation(
            function_name=function_name,
            operation_type='llm_call',
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time,
            context=context,
            input_size=prompt_length,
            output_size=response_length,
            success=success,
            error_message=error_message,
            model_name=model_name
        )
        
    def end_request_tracking(self) -> Dict[str, Any]:
        """End request tracking and generate comprehensive analysis"""
        if not self.request_start_time:
            return {}
            
        total_request_time = time.time() - self.request_start_time
        
        # Generate comprehensive analysis
        analysis = self._generate_performance_analysis(total_request_time)
        
        # Save to file if enabled
        if self.log_to_file:
            self._save_to_file(analysis)
        
        # Print detailed report
        self._print_detailed_report(analysis)
        
        return analysis
    
    def _generate_performance_analysis(self, total_request_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        
        # Basic metrics
        total_operations = len(self.entries)
        
        # LLM analysis
        llm_total_time = sum(entry.duration_ms for entry in self.llm_calls)
        llm_count = len(self.llm_calls)
        llm_avg_time = llm_total_time / llm_count if llm_count > 0 else 0
        
        # Function analysis
        func_total_time = sum(entry.duration_ms for entry in self.function_calls)
        func_count = len(self.function_calls)
        func_avg_time = func_total_time / func_count if func_count > 0 else 0
        
        # Cache analysis
        cache_total_time = sum(entry.duration_ms for entry in self.cache_operations)
        cache_count = len(self.cache_operations)
        cache_avg_time = cache_total_time / cache_count if cache_count > 0 else 0
        
        # Redis analysis
        redis_total_time = sum(entry.duration_ms for entry in self.redis_operations)
        redis_count = len(self.redis_operations)
        redis_avg_time = redis_total_time / redis_count if redis_count > 0 else 0
        
        # Calculate overhead
        accounted_time = llm_total_time + func_total_time + cache_total_time + redis_total_time
        overhead_time = (total_request_time * 1000) - accounted_time
        overhead_percentage = (overhead_time / (total_request_time * 1000)) * 100 if total_request_time > 0 else 0
        
        # Top slowest operations
        slowest_operations = sorted(self.entries, key=lambda x: x.duration_ms, reverse=True)[:10]
        
        # Failed operations
        failed_operations = [entry for entry in self.entries if not entry.success]
        
        return {
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat(),
            "total_request_time_ms": total_request_time * 1000,
            "total_operations": total_operations,
            
            # LLM metrics
            "llm_metrics": {
                "total_calls": llm_count,
                "total_time_ms": llm_total_time,
                "average_time_ms": llm_avg_time,
                "percentage_of_total": (llm_total_time / (total_request_time * 1000)) * 100 if total_request_time > 0 else 0,
                "calls": [asdict(entry) for entry in self.llm_calls]
            },
            
            # Function metrics
            "function_metrics": {
                "total_calls": func_count,
                "total_time_ms": func_total_time,
                "average_time_ms": func_avg_time,
                "percentage_of_total": (func_total_time / (total_request_time * 1000)) * 100 if total_request_time > 0 else 0
            },
            
            # Cache metrics
            "cache_metrics": {
                "total_calls": cache_count,
                "total_time_ms": cache_total_time,
                "average_time_ms": cache_avg_time,
                "percentage_of_total": (cache_total_time / (total_request_time * 1000)) * 100 if total_request_time > 0 else 0
            },
            
            # Redis metrics
            "redis_metrics": {
                "total_calls": redis_count,
                "total_time_ms": redis_total_time,
                "average_time_ms": redis_avg_time,
                "percentage_of_total": (redis_total_time / (total_request_time * 1000)) * 100 if total_request_time > 0 else 0
            },
            
            # Overhead analysis
            "overhead_analysis": {
                "accounted_time_ms": accounted_time,
                "unaccounted_time_ms": overhead_time,
                "overhead_percentage": overhead_percentage
            },
            
            # Performance insights
            "slowest_operations": [asdict(entry) for entry in slowest_operations],
            "failed_operations": [asdict(entry) for entry in failed_operations],
            "all_entries": [asdict(entry) for entry in self.entries]
        }
    
    def _print_detailed_report(self, analysis: Dict[str, Any]):
        """Print detailed performance report"""
        print(f"\n📊 === DETAILED LATENCY ANALYSIS: {self.request_id} ===")
        print(f"⏱️  Total Request Time: {analysis['total_request_time_ms']:.2f}ms")
        print(f"🔢 Total Operations: {analysis['total_operations']}")
        
        # LLM Analysis
        llm_metrics = analysis['llm_metrics']
        print(f"\n🤖 LLM PERFORMANCE:")
        print(f"   📞 Total Calls: {llm_metrics['total_calls']}")
        print(f"   ⏱️  Total Time: {llm_metrics['total_time_ms']:.2f}ms ({llm_metrics['percentage_of_total']:.1f}%)")
        print(f"   📊 Average Time: {llm_metrics['average_time_ms']:.2f}ms")
        
        # Function Analysis
        func_metrics = analysis['function_metrics']
        print(f"\n🔧 FUNCTION PERFORMANCE:")
        print(f"   📞 Total Calls: {func_metrics['total_calls']}")
        print(f"   ⏱️  Total Time: {func_metrics['total_time_ms']:.2f}ms ({func_metrics['percentage_of_total']:.1f}%)")
        print(f"   📊 Average Time: {func_metrics['average_time_ms']:.2f}ms")
        
        # Cache Analysis
        cache_metrics = analysis['cache_metrics']
        print(f"\n🗄️  CACHE PERFORMANCE:")
        print(f"   📞 Total Calls: {cache_metrics['total_calls']}")
        print(f"   ⏱️  Total Time: {cache_metrics['total_time_ms']:.2f}ms ({cache_metrics['percentage_of_total']:.1f}%)")
        print(f"   📊 Average Time: {cache_metrics['average_time_ms']:.2f}ms")
        
        # Redis Analysis
        redis_metrics = analysis['redis_metrics']
        print(f"\n🔴 REDIS PERFORMANCE:")
        print(f"   📞 Total Calls: {redis_metrics['total_calls']}")
        print(f"   ⏱️  Total Time: {redis_metrics['total_time_ms']:.2f}ms ({redis_metrics['percentage_of_total']:.1f}%)")
        print(f"   📊 Average Time: {redis_metrics['average_time_ms']:.2f}ms")
        
        # Overhead Analysis
        overhead = analysis['overhead_analysis']
        print(f"\n⚡ OVERHEAD ANALYSIS:")
        print(f"   ✅ Accounted Time: {overhead['accounted_time_ms']:.2f}ms")
        print(f"   ❓ Unaccounted Time: {overhead['unaccounted_time_ms']:.2f}ms ({overhead['overhead_percentage']:.1f}%)")
        
        # Top Slowest Operations
        print(f"\n🐌 TOP 5 SLOWEST OPERATIONS:")
        for i, op in enumerate(analysis['slowest_operations'][:5], 1):
            print(f"   {i}. {op['function_name']} ({op['operation_type']}): {op['duration_ms']:.2f}ms")
        
        # Failed Operations
        if analysis['failed_operations']:
            print(f"\n❌ FAILED OPERATIONS:")
            for op in analysis['failed_operations']:
                print(f"   • {op['function_name']}: {op['error_message']}")
        
        # Performance Recommendations
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
        if llm_metrics['percentage_of_total'] > 70:
            print(f"   🤖 LLM calls dominate ({llm_metrics['percentage_of_total']:.1f}%) - consider parallel processing")
        if llm_metrics['average_time_ms'] > 3000:
            print(f"   ⚠️  Average LLM call time is high ({llm_metrics['average_time_ms']:.0f}ms) - optimize prompts")
        if overhead['overhead_percentage'] > 30:
            print(f"   🔧 High overhead ({overhead['overhead_percentage']:.1f}%) - profile non-tracked operations")
        if cache_metrics['total_calls'] == 0:
            print(f"   💾 No cache operations detected - implement caching")
        
        print(f"🏁 === END LATENCY ANALYSIS ===\n")
    
    def _save_to_file(self, analysis: Dict[str, Any]):
        """Save analysis to JSON file"""
        try:
            import os
            os.makedirs("logs/latency", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/latency/interview_latency_{self.request_id}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"💾 Latency analysis saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save latency analysis: {e}")

# Global logger instance
detailed_logger = DetailedLatencyLogger()

# Convenience decorators
def time_function(context: str = ""):
    """Decorator for timing regular functions"""
    return detailed_logger.function_timer(operation_type="function", context=context)

def time_llm_function(context: str = ""):
    """Decorator for timing LLM-related functions"""
    return detailed_logger.function_timer(operation_type="llm_call", context=context)

def time_cache_function(context: str = ""):
    """Decorator for timing cache operations"""
    return detailed_logger.function_timer(operation_type="cache_lookup", context=context)

def time_redis_function(context: str = ""):
    """Decorator for timing Redis operations"""  
    return detailed_logger.function_timer(operation_type="redis_operation", context=context)