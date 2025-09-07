"""
Dynamic System Adapter for Cell Evaluation

This module provides intelligent system adaptation and configuration optimization
for multiprocessing workloads in the cell_eval package.

Author: Enhanced Multiprocessing Integration
Version: 1.0.0
"""

import os
import sys
import time
import psutil
import platform
import threading
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Try to import optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False

# === PDEX SPEED OPTIMIZATION START ===
# === ECHTE PARALLELISIERUNG SETUP ===
import os
import multiprocessing as mp
import psutil
#from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor


# Intelligente Ressourcen-Erkennung
def get_optimal_workers():
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().available / (1024**3)
    
    if memory_gb > 16:  # Viel RAM
        return min(cpu_count, 8)
    elif memory_gb > 8:  # Mittlerer RAM
        return min(cpu_count, 4)
    else:  # Wenig RAM
        return min(cpu_count, 2)

OPTIMAL_WORKERS = get_optimal_workers()
OPTIMAL_THREADS = max(1, OPTIMAL_WORKERS // 2)

# Multi-Threading Environment
os.environ['OMP_NUM_THREADS'] = str(OPTIMAL_THREADS)
os.environ['MKL_NUM_THREADS'] = str(OPTIMAL_THREADS) 
os.environ['NUMEXPR_NUM_THREADS'] = str(OPTIMAL_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(OPTIMAL_THREADS)

print(f"🚀 Parallelisierung: {OPTIMAL_WORKERS} Workers, {OPTIMAL_THREADS} Threads/Worker")
# === ENDE SETUP ===
# Set multiprocessing method
import platform
try:
    if platform.system() in ['Linux', 'Darwin']:  # Linux/Mac
        mp.set_start_method('fork', force=True)
        print("🚀 Fork method aktiviert (schnell + stabil)")
    else:  # Windows
        mp.set_start_method('spawn', force=True)
        print("⚠️ Spawn method (Windows)")
except RuntimeError as e:
    current_method = mp.get_start_method()
    print(f"ℹ️ Start method bereits gesetzt: {current_method}")
# PDEX-safe progress bars
def _pdex_tqdm_wrapper(original_tqdm):
    def wrapper(*args, **kwargs):
        if mp.current_process().name != 'MainProcess':
            kwargs['disable'] = True
        return original_tqdm(*args, **kwargs)
    return wrapper

try:
    import tqdm
    if not hasattr(tqdm.tqdm, '_pdex_wrapped'):
        tqdm.tqdm._original = tqdm.tqdm.__init__
        tqdm.tqdm.__init__ = _pdex_tqdm_wrapper(tqdm.tqdm._original)
        tqdm.tqdm._pdex_wrapped = True
except ImportError:
    pass
# === PDEX SPEED OPTIMIZATION END ===


@dataclass
class SystemSpecs:
    """System specifications container"""
    cpu_count: int
    cpu_freq: float
    memory_total: int
    memory_available: int
    platform_system: str
    platform_machine: str
    python_version: str
    cpu_brand: Optional[str] = None
    cpu_features: List[str] = field(default_factory=list)
    load_average: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.load_average is None and hasattr(os, 'getloadavg'):
            try:
                self.load_average = os.getloadavg()[0]
            except:
                self.load_average = 0.0


@dataclass
class WorkloadProfile:
    """Workload profile for optimization"""
    name: str
    cpu_intensive: bool = False
    memory_intensive: bool = False
    io_intensive: bool = False
    preferred_process_count: Optional[int] = None
    preferred_chunk_size: Optional[int] = None
    timeout_multiplier: float = 1.0
    description: str = ""


class DynamicSystemAdapter:
    """
    Intelligent system adapter that optimizes multiprocessing configuration
    based on current system state and workload characteristics.
    """
    
    # Predefined workload profiles
    WORKLOAD_PROFILES = {
        'general': WorkloadProfile(
            name='general',
            description='General purpose workload with balanced resource usage'
        ),
        'cpu_intensive': WorkloadProfile(
            name='cpu_intensive',
            cpu_intensive=True,
            preferred_process_count=None,  # Will use CPU count
            preferred_chunk_size=1,
            timeout_multiplier=2.0,
            description='CPU-intensive tasks requiring maximum parallel processing'
        ),
        'memory_intensive': WorkloadProfile(
            name='memory_intensive',
            memory_intensive=True,
            preferred_process_count=None,  # Will be calculated based on memory
            preferred_chunk_size=10,
            timeout_multiplier=1.5,
            description='Memory-intensive tasks requiring careful memory management'
        ),
        'io_intensive': WorkloadProfile(
            name='io_intensive',
            io_intensive=True,
            preferred_process_count=None,  # Will use higher count for I/O waiting
            preferred_chunk_size=50,
            timeout_multiplier=3.0,
            description='I/O-intensive tasks that benefit from higher concurrency'
        ),
        'mixed': WorkloadProfile(
            name='mixed',
            cpu_intensive=True,
            memory_intensive=True,
            preferred_chunk_size=5,
            timeout_multiplier=1.8,
            description='Mixed workload with both CPU and memory requirements'
        )
    }
    
    def __init__(self, 
                 update_interval: float = 30.0,
                 enable_monitoring: bool = True,
                 cache_duration: float = 60.0):
        """
        Initialize the dynamic system adapter
        
        Args:
            update_interval: How often to update system metrics (seconds)
            enable_monitoring: Enable continuous system monitoring
            cache_duration: How long to cache system specs (seconds)
        """
        self.update_interval = update_interval
        self.enable_monitoring = enable_monitoring
        self.cache_duration = cache_duration
        
        # System state
        self._system_specs: Optional[SystemSpecs] = None
        self._last_update: float = 0
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []
        self._current_load: float = 0.0
        
        # Initialize
        self._update_system_specs()
        
        if self.enable_monitoring:
            self._start_monitoring()
    
    def _update_system_specs(self) -> None:
        """Update system specifications"""
        current_time = time.time()
        
        # Check cache validity
        if (self._system_specs is not None and 
            current_time - self._last_update < self.cache_duration):
            return
        
        try:
            # Get basic system info
            cpu_count = 1 #mp.cpu_count()
            memory = psutil.virtual_memory()
            
            # Get CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                freq = cpu_freq.current if cpu_freq else 0.0
            except:
                freq = 0.0
            
            # Get CPU brand if available
            cpu_brand = None
            cpu_features = []
            if HAS_CPUINFO:
                try:
                    info = cpuinfo.get_cpu_info()
                    cpu_brand = info.get('brand_raw', info.get('brand'))
                    cpu_features = info.get('flags', [])
                except:
                    pass
            
            # Get load average
            load_avg = None
            if hasattr(os, 'getloadavg'):
                try:
                    load_avg = os.getloadavg()[0]
                except:
                    pass
            
            self._system_specs = SystemSpecs(
                cpu_count=cpu_count,
                cpu_freq=freq,
                memory_total=memory.total,
                memory_available=memory.available,
                platform_system=platform.system(),
                platform_machine=platform.machine(),
                python_version=sys.version,
                cpu_brand=cpu_brand,
                cpu_features=cpu_features,
                load_average=load_avg
            )
            
            self._last_update = current_time
            
        except Exception as e:
            warnings.warn(f"Failed to update system specs: {e}")
    
    def _start_monitoring(self) -> None:
        """Start system monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop"""
        while not self._stop_monitoring.wait(self.update_interval):
            try:
                self._update_system_specs()
                self._current_load = psutil.cpu_percent(interval=1)
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def get_system_specs(self) -> SystemSpecs:
        """Get current system specifications"""
        self._update_system_specs()
        return self._system_specs
    
    def get_current_load(self) -> float:
        """Get current system load"""
        if self.enable_monitoring:
            return self._current_load
        else:
            return psutil.cpu_percent(interval=1)
    
    def calculate_optimal_process_count(self, 
                                      workload_profile: WorkloadProfile,
                                      max_processes: Optional[int] = None) -> int:
        """
        Calculate optimal process count based on workload and system state
        
        Args:
            workload_profile: Workload characteristics
            max_processes: Maximum allowed processes
        
        Returns:
            Optimal process count
        """
        specs = self.get_system_specs()
        base_count = specs.cpu_count
        
        # Apply workload-specific adjustments
        if workload_profile.preferred_process_count is not None:
            optimal_count = workload_profile.preferred_process_count
        elif workload_profile.cpu_intensive:
            # CPU-intensive: use CPU count
            optimal_count = base_count
        elif workload_profile.memory_intensive:
            # Memory-intensive: reduce based on available memory
            # Assume each process needs ~100MB minimum
            memory_gb = specs.memory_available / (1024**3)
            memory_based_count = max(1, int(memory_gb * 10))
            optimal_count = min(base_count, memory_based_count)
        elif workload_profile.io_intensive:
            # I/O-intensive: can use more processes due to waiting
            optimal_count = min(base_count * 2, 32)  # Cap at 32
        else:
            # General workload
            optimal_count = max(1, base_count - 1)  # Leave one core free
        
        # Adjust based on current system load
        current_load = self.get_current_load()
        if current_load > 80:
            optimal_count = max(1, optimal_count // 2)
        elif current_load > 60:
            optimal_count = max(1, int(optimal_count * 0.75))
        
        # Apply maximum limit
        if max_processes is not None:
            optimal_count = min(optimal_count, max_processes)
        
        return max(1, optimal_count)
    
    def calculate_optimal_chunk_size(self, 
                                   workload_profile: WorkloadProfile,
                                   total_items: int,
                                   process_count: int) -> int:
        """
        Calculate optimal chunk size for workload distribution
        
        Args:
            workload_profile: Workload characteristics
            total_items: Total number of items to process
            process_count: Number of processes
        
        Returns:
            Optimal chunk size
        """
        if workload_profile.preferred_chunk_size is not None:
            base_chunk_size = workload_profile.preferred_chunk_size
        else:
            # Default calculation: distribute evenly with some overhead
            base_chunk_size = max(1, total_items // (process_count * 4))
        
        # Adjust based on workload characteristics
        if workload_profile.cpu_intensive:
            # Smaller chunks for better load balancing
            chunk_size = max(1, base_chunk_size // 2)
        elif workload_profile.memory_intensive:
            # Larger chunks to reduce memory overhead
            chunk_size = base_chunk_size * 2
        elif workload_profile.io_intensive:
            # Medium chunks for I/O efficiency
            chunk_size = base_chunk_size
        else:
            chunk_size = base_chunk_size
        
        # Ensure reasonable bounds
        chunk_size = max(1, min(chunk_size, total_items // 2))
        
        return chunk_size
    
    def get_optimal_config(self, 
                          workload_type: str = 'general',
                          total_items: Optional[int] = None,
                          max_processes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get optimal configuration for a specific workload
        
        Args:
            workload_type: Type of workload ('general', 'cpu_intensive', etc.)
            total_items: Total number of items to process
            max_processes: Maximum allowed processes
        
        Returns:
            Dictionary with optimal configuration
        """
        # Get workload profile
        profile = self.WORKLOAD_PROFILES.get(workload_type)
        if profile is None:
            warnings.warn(f"Unknown workload type: {workload_type}, using 'general'")
            profile = self.WORKLOAD_PROFILES['general']
        
        # Calculate optimal settings
        process_count = self.calculate_optimal_process_count(profile, max_processes)
        
        chunk_size = 1
        if total_items is not None:
            chunk_size = self.calculate_optimal_chunk_size(profile, total_items, process_count)
        
        # Calculate timeout
        base_timeout = 30.0  # Base timeout in seconds
        timeout = base_timeout * profile.timeout_multiplier
        
        config = {
            'process_count': process_count,
            'chunk_size': chunk_size,
            'timeout': timeout,
            'workload_type': workload_type,
            'workload_profile': profile,
            'system_load': self.get_current_load(),
            'timestamp': time.time()
        }
        
        return config
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current optimal configuration for general workload"""
        return self.get_optimal_config('general')
    
    def benchmark_system(self, 
                        duration: float = 5.0,
                        workload_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark system performance for different workload types
        
        Args:
            duration: Benchmark duration in seconds
            workload_types: List of workload types to benchmark
        
        Returns:
            Benchmark results
        """
        if workload_types is None:
            workload_types = ['general', 'cpu_intensive', 'memory_intensive']
        
        results = {
            'system_specs': self.get_system_specs().__dict__,
            'benchmark_duration': duration,
            'workload_results': {},
            'timestamp': time.time()
        }
        
        def cpu_benchmark():
            """Simple CPU benchmark"""
            start = time.time()
            count = 0
            while time.time() - start < duration:
                count += 1
                # Simple computation
                _ = sum(i * i for i in range(1000))
            return count
        
        # Benchmark each workload type
        for workload_type in workload_types:
            config = self.get_optimal_config(workload_type)
            
            # Run benchmark
            start_time = time.time()
            
            try:
                with ThreadPoolExecutor(max_workers=config['process_count']) as executor:
                    futures = [executor.submit(cpu_benchmark) for _ in range(config['process_count'])]
                    benchmark_results = [f.result() for f in futures]
                
                end_time = time.time()
                
                results['workload_results'][workload_type] = {
                    'config': config,
                    'execution_time': end_time - start_time,
                    'operations_per_second': sum(benchmark_results) / (end_time - start_time),
                    'individual_results': benchmark_results
                }
                
            except Exception as e:
                results['workload_results'][workload_type] = {
                    'config': config,
                    'error': str(e)
                }
        
        return results
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_monitoring()


# Convenience functions
def get_optimal_config(workload_type: str = 'general', 
                      total_items: Optional[int] = None,
                      max_processes: Optional[int] = None) -> Dict[str, Any]:
    """
    Get optimal configuration for a workload type
    
    Args:
        workload_type: Type of workload
        total_items: Total number of items to process
        max_processes: Maximum allowed processes
    
    Returns:
        Optimal configuration dictionary
    """
    adapter = DynamicSystemAdapter(enable_monitoring=False)
    return adapter.get_optimal_config(workload_type, total_items, max_processes)


def print_system_info():
    """Print detailed system information"""
    adapter = DynamicSystemAdapter(enable_monitoring=False)
    specs = adapter.get_system_specs()
    
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Platform: {specs.platform_system} {specs.platform_machine}")
    print(f"CPU Count: {specs.cpu_count}")
    if specs.cpu_brand:
        print(f"CPU Brand: {specs.cpu_brand}")
    if specs.cpu_freq > 0:
        print(f"CPU Frequency: {specs.cpu_freq:.2f} MHz")
    print(f"Memory Total: {specs.memory_total / (1024**3):.2f} GB")
    print(f"Memory Available: {specs.memory_available / (1024**3):.2f} GB")
    if specs.load_average is not None:
        print(f"Load Average: {specs.load_average:.2f}")
    print(f"Python Version: {specs.python_version.split()[0]}")
    
    current_load = adapter.get_current_load()
    print(f"Current CPU Load: {current_load:.1f}%")
    
    print("\\n📊 WORKLOAD RECOMMENDATIONS")
    print("=" * 50)
    for workload_type in ['general', 'cpu_intensive', 'memory_intensive', 'io_intensive']:
        config = adapter.get_optimal_config(workload_type)
        print(f"{workload_type.upper()}: {config['process_count']} processes, "
              f"chunk_size={config['chunk_size']}, timeout={config['timeout']:.1f}s")


def benchmark_system_performance(duration: float = 5.0) -> Dict[str, Any]:
    """
    Benchmark system performance
    
    Args:
        duration: Benchmark duration in seconds
    
    Returns:
        Benchmark results
    """
    adapter = DynamicSystemAdapter(enable_monitoring=False)
    return adapter.benchmark_system(duration)


if __name__ == "__main__":
    # Demo when run directly
    print_system_info()
    print("\\n🔬 RUNNING BENCHMARK...")
    results = benchmark_system_performance(3.0)
    print(f"Benchmark completed in {results['benchmark_duration']} seconds")
    for workload, result in results['workload_results'].items():
        if 'error' not in result:
            print(f"{workload}: {result['operations_per_second']:.0f} ops/sec")