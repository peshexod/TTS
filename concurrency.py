#!/usr/bin/env python3
"""
Adaptive Concurrency Modifier for RunPod Serverless - Coqui TTS
Monitors GPU/CPU usage and adjusts concurrency dynamically
"""

import os

# Try to import monitoring libraries
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configuration
GPU_THRESHOLD_HIGH = 80  # At 80% GPU - reduce jobs
GPU_THRESHOLD_LOW = 50  # At 50% GPU - allow more jobs
CPU_THRESHOLD_HIGH = 80  # At 80% CPU - reduce jobs

# Limits
MAX_CONCURRENCY = 20
MIN_CONCURRENCY = 1

# Initialize NVML
if NVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
        GPU_COUNT = pynvml.nvmlDeviceGetCount()
    except Exception as e:
        print(f"Warning: NVML init failed: {e}")
        GPU_COUNT = 0
else:
    GPU_COUNT = 0


def get_gpu_utilization():
    """Get current GPU utilization percentage"""
    if not NVML_AVAILABLE or GPU_COUNT == 0:
        return 0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except Exception:
        return 0


def get_cpu_utilization():
    """Get current CPU utilization percentage"""
    if not PSUTIL_AVAILABLE:
        return 0
    
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0


def get_memory_usage():
    """Get current memory usage percentage"""
    if not PSUTIL_AVAILABLE:
        return 0
    
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0


def adjust_concurrency(current_concurrency):
    """
    RunPod concurrency modifier callback.
    Dynamically adjusts concurrency based on GPU/CPU load.
    
    Args:
        current_concurrency: Current number of running jobs
    
    Returns:
        int: New concurrency level to set
    """
    gpu_util = get_gpu_utilization()
    cpu_util = get_cpu_utilization()
    mem_util = get_memory_usage()
    
    # Check if overloaded
    is_overloaded = (gpu_util >= GPU_THRESHOLD_HIGH or 
                     cpu_util >= CPU_THRESHOLD_HIGH or 
                     mem_util >= 80)
    
    if is_overloaded and current_concurrency > MIN_CONCURRENCY:
        # High load - reduce concurrency
        new_level = current_concurrency - 1
        print(f"[Concurrency] High load: GPU={gpu_util}%, CPU={cpu_util}%, MEM={mem_util}%. Reducing to {new_level}")
        return new_level
    
    elif current_concurrency < MAX_CONCURRENCY:
        # Low load - increase concurrency
        new_level = current_concurrency + 1
        print(f"[Concurrency] Low load: GPU={gpu_util}%, CPU={cpu_util}%, MEM={mem_util}%. Increasing to {new_level}")
        return new_level
    
    # Keep current concurrency
    return current_concurrency
