"""
Quantization Benchmarking Utilities

This module provides utilities for benchmarking different quantization techniques
(FP16, 8-bit, 4-bit) and speculative decoding for Large Language Models.
"""

import torch
import psutil
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results"""
    model_name: str
    quantization: str  # 'fp16', '8bit', '4bit'
    speculative_decoding: bool
    
    # Performance Metrics
    inference_time: float  # seconds
    tokens_per_second: float
    speedup: float  # relative to baseline
    
    # Memory Metrics
    vram_used_mb: float
    vram_savings_percent: float
    
    # Quality Metrics
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Cost Metrics
    training_time_per_epoch: Optional[float] = None
    estimated_cost_per_epoch: Optional[float] = None  # in USD
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)


class GPUMonitor:
    """Monitor GPU memory usage and utilization"""
    
    def __init__(self):
        self.available = PYNVML_AVAILABLE
        if self.available:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.available = False
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB"""
        if not self.available:
            return {"used": 0.0, "total": 0.0, "free": 0.0}
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "used": info.used / 1024**2,
                "total": info.total / 1024**2,
                "free": info.free / 1024**2,
            }
        except:
            return {"used": 0.0, "total": 0.0, "free": 0.0}
    
    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        if not self.available:
            return 0.0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except:
            return 0.0
    
    def __del__(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class MemoryTracker:
    """Track memory usage during model operations"""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.start_memory = None
        self.peak_memory = None
    
    def start(self):
        """Start tracking memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.start_memory = self.gpu_monitor.get_gpu_memory_usage()
        self.peak_memory = self.start_memory["used"]
    
    def update_peak(self):
        """Update peak memory usage"""
        current = self.gpu_monitor.get_gpu_memory_usage()["used"]
        if current > self.peak_memory:
            self.peak_memory = current
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        current = self.gpu_monitor.get_gpu_memory_usage()
        
        if torch.cuda.is_available():
            cuda_peak = torch.cuda.max_memory_allocated() / 1024**2
            self.peak_memory = max(self.peak_memory, cuda_peak)
        
        return {
            "start_mb": self.start_memory["used"] if self.start_memory else 0,
            "current_mb": current["used"],
            "peak_mb": self.peak_memory,
            "allocated_mb": self.peak_memory - (self.start_memory["used"] if self.start_memory else 0),
        }


class InferenceTimer:
    """Time inference operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.token_count = 0
    
    def start(self):
        """Start timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def stop(self, token_count: int = 0):
        """Stop timing and record token count"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.token_count = token_count
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def get_tokens_per_second(self) -> float:
        """Get tokens per second"""
        elapsed = self.get_elapsed_time()
        if elapsed > 0 and self.token_count > 0:
            return self.token_count / elapsed
        return 0.0


def calculate_speedup(baseline_time: float, current_time: float) -> float:
    """Calculate speedup relative to baseline"""
    if current_time > 0:
        return baseline_time / current_time
    return 0.0


def calculate_vram_savings(baseline_vram: float, current_vram: float) -> float:
    """Calculate VRAM savings percentage"""
    if baseline_vram > 0:
        return ((baseline_vram - current_vram) / baseline_vram) * 100
    return 0.0


def estimate_training_cost(
    time_per_epoch: float,
    gpu_type: str = "A100",
    cost_per_hour: Optional[float] = None
) -> float:
    """
    Estimate training cost per epoch
    
    Default costs (per hour):
    - A100 (40GB): $1.10
    - A100 (80GB): $1.40
    - V100: $0.80
    - T4: $0.35
    """
    default_costs = {
        "A100": 1.10,
        "A100-80GB": 1.40,
        "V100": 0.80,
        "T4": 0.35,
    }
    
    if cost_per_hour is None:
        cost_per_hour = default_costs.get(gpu_type, 1.10)
    
    hours = time_per_epoch / 3600
    return hours * cost_per_hour


def save_benchmark_results(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results to JSON file"""
    data = [result.to_dict() for result in results]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_benchmark_results(filepath: str) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [BenchmarkResult(**item) for item in data]


def print_benchmark_summary(result: BenchmarkResult):
    """Print a formatted summary of benchmark results"""
    print(f"\n{'='*60}")
    print(f"Model: {result.model_name}")
    print(f"Quantization: {result.quantization}")
    print(f"Speculative Decoding: {result.speculative_decoding}")
    print(f"{'-'*60}")
    print(f"Inference Time: {result.inference_time:.2f}s")
    print(f"Tokens/sec: {result.tokens_per_second:.2f}")
    print(f"Speedup: {result.speedup:.2f}x")
    print(f"VRAM Used: {result.vram_used_mb:.2f} MB")
    print(f"VRAM Savings: {result.vram_savings_percent:.2f}%")
    
    if result.perplexity is not None:
        print(f"Perplexity: {result.perplexity:.2f}")
    if result.accuracy is not None:
        print(f"Accuracy: {result.accuracy:.2f}%")
    if result.estimated_cost_per_epoch is not None:
        print(f"Est. Cost/Epoch: ${result.estimated_cost_per_epoch:.2f}")
    print(f"{'='*60}\n")
