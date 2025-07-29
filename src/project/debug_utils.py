"""
Debug and monitoring utilities for Neural Architecture Search experiments.
Provides consistent logging, progress tracking, and performance monitoring.
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from datetime import datetime

class ProgressTracker:
    """Tracks progress and performance metrics during experiments."""
    
    def __init__(self, name: str, total_steps: Optional[int] = None):
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.step_times = []
        
    def update(self, step: Optional[int] = None, message: str = ""):
        """Update progress tracker."""
        current_time = time.time()
        
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        # Calculate timing
        step_time = current_time - self.last_update
        self.step_times.append(step_time)
        self.last_update = current_time
        
        # Calculate ETA
        if self.total_steps and self.current_step > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_step_time * remaining_steps
            eta_str = f" | ETA: {eta_seconds/60:.1f}min"
            progress_pct = (self.current_step / self.total_steps) * 100
            progress_str = f" ({progress_pct:.1f}%)"
        else:
            eta_str = ""
            progress_str = ""
        
        elapsed = current_time - self.start_time
        print(f"⏱️  {self.name} - Step {self.current_step}{progress_str} | "
              f"Elapsed: {elapsed/60:.1f}min{eta_str} | {message}")
    
    def finish(self, message: str = ""):
        """Mark progress as finished."""
        total_time = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        print(f"✅ {self.name} completed!")
        print(f"   📊 Total time: {total_time/60:.1f}min")
        print(f"   📊 Total steps: {self.current_step}")
        print(f"   📊 Avg step time: {avg_step_time:.2f}s")
        if message:
            print(f"   📝 {message}")

class SystemMonitor:
    """Monitors system resources during experiments."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get current system information."""
        info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # GPU info if available
        if torch.cuda.is_available():
            info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            info['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        
        return info
    
    @staticmethod
    def print_system_status():
        """Print current system status."""
        info = SystemMonitor.get_system_info()
        
        print(f"💻 System Status:")
        print(f"   🖥️  CPU: {info['cpu_percent']:.1f}%")
        print(f"   🧠 RAM: {info['memory_percent']:.1f}% ({info['memory_available_gb']:.1f}GB available)")
        
        if 'gpu_memory_allocated_mb' in info:
            gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0
            gpu_usage_pct = (info['gpu_memory_allocated_mb'] / gpu_total_mb * 100) if gpu_total_mb > 0 else 0
            print(f"   🎮 GPU Memory: {info['gpu_memory_allocated_mb']:.0f}MB allocated / {gpu_total_mb:.0f}MB total ({gpu_usage_pct:.1f}%)")
            print(f"   🎮 GPU Reserved: {info['gpu_memory_reserved_mb']:.0f}MB")
        
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache if CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cache cleared")

class DebugLogger:
    """Enhanced logging with different verbosity levels."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level.upper()
        self.levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        self.current_level = self.levels.get(self.level, 1)
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        return self.levels.get(level.upper(), 1) >= self.current_level
    
    def debug(self, message: str):
        """Log debug message."""
        if self._should_log("DEBUG"):
            print(f"🔍 [{self.name}] DEBUG: {message}")
    
    def info(self, message: str):
        """Log info message."""
        if self._should_log("INFO"):
            print(f"ℹ️  [{self.name}] {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        if self._should_log("WARNING"):
            print(f"⚠️  [{self.name}] WARNING: {message}")
    
    def error(self, message: str):
        """Log error message."""
        if self._should_log("ERROR"):
            print(f"❌ [{self.name}] ERROR: {message}")
    
    def success(self, message: str):
        """Log success message."""
        if self._should_log("INFO"):
            print(f"✅ [{self.name}] {message}")

class PerformanceProfiler:
    """Profiles performance of different operations."""
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
    
    def start_timer(self, name: str):
        """Start a timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and return elapsed time."""
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'active_timers': list(self.timers.keys()),
            'counters': self.counters.copy()
        }
    
    def print_stats(self):
        """Print performance statistics."""
        print("📊 Performance Stats:")
        for name, count in self.counters.items():
            print(f"   📈 {name}: {count}")
        
        if self.timers:
            print("   ⏱️  Active timers:", ", ".join(self.timers.keys()))

# Global instances for easy access
global_profiler = PerformanceProfiler()
system_monitor = SystemMonitor()

def create_logger(name: str, level: str = "INFO") -> DebugLogger:
    """Create a debug logger instance."""
    return DebugLogger(name, level)

def create_progress_tracker(name: str, total_steps: Optional[int] = None) -> ProgressTracker:
    """Create a progress tracker instance."""
    return ProgressTracker(name, total_steps)
