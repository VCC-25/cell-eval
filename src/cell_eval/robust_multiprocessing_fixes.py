"""
🔧 ROBUST MULTIPROCESSING FIXES
===============================

Robuste Multiprocessing-Lösungen für cell_load Framework
Behebt häufige Probleme mit multiprocessing in Python:
- Memory leaks
- Process hanging
- Resource cleanup
- Cross-platform compatibility
- GPU memory management

Author: Enhanced Cell Load Framework
"""

import os
import sys
import time
import signal
import logging
import warnings
import threading
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from functools import wraps, partial
from contextlib import contextmanager
import gc
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue

# Try to import psutil, fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available, process monitoring disabled")

# Setup logging
logger = logging.getLogger(__name__)

# === PDEX SPEED OPTIMIZATION START ===
#import os
import multiprocessing as mp

# === ECHTE PARALLELISIERUNG SETUP ===
import os
import multiprocessing as mp
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

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
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

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

# Global configuration
ROBUST_MP_CONFIG = {
    'max_memory_per_process': 8 * 1024 * 1024 * 1024,  # 8GB
    'process_timeout': 3600,  # 1 hour
    'cleanup_interval': 60,   # 1 minute
    'max_retries': 3,
    'enable_memory_monitoring': True,
    'enable_gpu_cleanup': True,
    'force_gc_interval': 100,  # Every 100 operations
}


class ProcessMonitor:
    """
    🔍 Process Monitor für robustes multiprocessing
    
    Überwacht Prozesse auf:
    - Memory usage
    - CPU usage  
    - GPU memory (wenn verfügbar)
    - Process health
    """
    
    def __init__(self, max_memory_mb: int = 8192):
        self.max_memory_mb = max_memory_mb
        self.processes = {}
        self.monitoring = False
        self.monitor_thread = None
        self.enabled = PSUTIL_AVAILABLE
        
    def register_process(self, pid: int, name: str = "unknown"):
        """Registriere Prozess für monitoring"""
        if not self.enabled:
            return
            
        try:
            process = psutil.Process(pid)
            self.processes[pid] = {
                'process': process,
                'name': name,
                'start_time': time.time(),
                'memory_warnings': 0
            }
            logger.debug(f"📊 Registered process {pid} ({name})")
        except (psutil.NoSuchProcess, NameError):
            logger.warning(f"⚠️ Process {pid} not found for registration")
    
    def check_process_health(self, pid: int) -> Dict[str, Any]:
        """Prüfe process health"""
        if not self.enabled or pid not in self.processes:
            return {'status': 'unknown', 'error': 'Process not registered or monitoring disabled'}
        
        try:
            proc_info = self.processes[pid]
            process = proc_info['process']
            
            # Memory check
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU check
            cpu_percent = process.cpu_percent()
            
            # Status check
            status = process.status()
            
            health_info = {
                'status': status,
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'runtime': time.time() - proc_info['start_time'],
                'memory_exceeded': memory_mb > self.max_memory_mb,
                'healthy': status in ['running', 'sleeping'] and memory_mb < self.max_memory_mb
            }
            
            # Memory warning
            if memory_mb > self.max_memory_mb:
                proc_info['memory_warnings'] += 1
                if proc_info['memory_warnings'] > 3:
                    logger.warning(f"⚠️ Process {pid} consistently exceeding memory limit: {memory_mb:.1f}MB")
            
            return health_info
            
        except (psutil.NoSuchProcess, NameError):
            return {'status': 'terminated', 'error': 'Process no longer exists'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def cleanup_process(self, pid: int, force: bool = False):
        """Cleanup process"""
        if not self.enabled or pid not in self.processes:
            return
            
        try:
            process = self.processes[pid]['process']
            if process.is_running():
                if force:
                    process.kill()
                    logger.info(f"🔥 Force killed process {pid}")
                else:
                    process.terminate()
                    logger.info(f"🛑 Terminated process {pid}")
            del self.processes[pid]
        except Exception as e:
            logger.error(f"❌ Failed to cleanup process {pid}: {e}")
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring"""
        if not self.enabled or self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    for pid in list(self.processes.keys()):
                        health = self.check_process_health(pid)
                        
                        if not health.get('healthy', False):
                            logger.warning(f"⚠️ Unhealthy process detected: {pid}")
                            
                            if health.get('memory_exceeded', False):
                                logger.warning(f"💾 Process {pid} memory exceeded, considering cleanup")
                        
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"❌ Monitor loop error: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Process monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Process monitoring stopped")


# Global process monitor
_process_monitor = ProcessMonitor()


class RobustProcessPool:
    """
    🛡️ Robust Process Pool mit erweiterten Sicherheitsfeatures
    
    Features:
    - Automatic process health monitoring
    - Memory leak prevention
    - GPU memory cleanup
    - Graceful error handling
    - Process recycling
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_memory_per_process: int = 8192,  # MB
        process_timeout: int = 3600,  # seconds
        enable_monitoring: bool = True,
        recycle_after: int = 100,  # tasks per process
        **kwargs
    ):
        
        self.max_workers = max_workers or min(2, 1 or 4)
        self.max_memory_per_process = max_memory_per_process
        self.process_timeout = process_timeout
        self.enable_monitoring = enable_monitoring and PSUTIL_AVAILABLE
        self.recycle_after = recycle_after
        
        self.pool = None
        self.task_counts = {}
        self.active_futures = set()
        
        logger.info(f"🛡️ Initializing RobustProcessPool with {self.max_workers} workers")
    
    def __enter__(self):
        self._start_pool()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pool()
    
    def _start_pool(self):
        """Start the process pool"""
        try:
            # Set multiprocessing start method
            '''if hasattr(mp, 'set_start_method'):
                try:
                    mp.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass  # Already set
            '''
            mp.set_start_method('spawn', force=True)
            self.pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context('spawn') if hasattr(mp, 'get_context') else None
            )
            
            if self.enable_monitoring:
                _process_monitor.start_monitoring()
            
            logger.info(f"✅ RobustProcessPool started with {self.max_workers} workers")
            
        except Exception as e:
            logger.error(f"❌ Failed to start process pool: {e}")
            raise
    
    def _cleanup_pool(self):
        """Cleanup the process pool"""
        try:
            # Cancel active futures
            for future in list(self.active_futures):
                future.cancel()
            
            # Shutdown pool
            if self.pool:
                self.pool.shutdown(wait=True)
                logger.info("🛑 Process pool shutdown complete")
            
            # Stop monitoring
            if self.enable_monitoring:
                _process_monitor.stop_monitoring()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Error during pool cleanup: {e}")
    
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit task with robust error handling"""
        if not self.pool:
            raise RuntimeError("Process pool not initialized")
        
        # Wrap function for robust execution
        wrapped_fn = self._wrap_function(fn)
        
        try:
            future = self.pool.submit(wrapped_fn, *args, **kwargs)
            self.active_futures.add(future)
            
            # Remove from active when done
            def cleanup_future(fut):
                self.active_futures.discard(fut)
            
            future.add_done_callback(cleanup_future)
            
            return future
            
        except Exception as e:
            logger.error(f"❌ Failed to submit task: {e}")
            raise
    
    def map(self, fn: Callable, iterable, timeout: Optional[int] = None, chunksize: int = 1):
        """Robust map with monitoring"""
        if not self.pool:
            raise RuntimeError("Process pool not initialized")
        
        wrapped_fn = self._wrap_function(fn)
        timeout = timeout or self.process_timeout
        
        try:
            # Submit all tasks
            futures = [self.submit(wrapped_fn, item) for item in iterable]
            
            # Collect results with timeout
            results = []
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Task failed: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Map operation failed: {e}")
            raise
    
    def _wrap_function(self, fn: Callable) -> Callable:
        """Wrap function for robust execution"""
        
        @wraps(fn)
        def wrapped_function(*args, **kwargs):
            pid = os.getpid()
            
            try:
                # Register process for monitoring
                _process_monitor.register_process(pid, fn.__name__)
                
                # Setup signal handlers
                self._setup_signal_handlers()
                
                # Execute function
                result = fn(*args, **kwargs)
                
                # Cleanup
                self._cleanup_after_execution()
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Function execution failed in process {pid}: {e}")
                logger.error(traceback.format_exc())
                
                # Emergency cleanup
                self._emergency_cleanup()
                
                raise
        
        return wrapped_function
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, cleaning up...")
            self._emergency_cleanup()
            sys.exit(1)
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Not in main thread
            pass
    
    def _cleanup_after_execution(self):
        """Cleanup after function execution"""
        try:
            # Force garbage collection
            gc.collect()
            
            # GPU cleanup if available
            if ROBUST_MP_CONFIG['enable_gpu_cleanup']:
                self._cleanup_gpu_memory()
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup"""
        try:
            gc.collect()
            self._cleanup_gpu_memory()
        except Exception:
            pass
    
    def _cleanup_gpu_memory(self):
        """Cleanup GPU memory if available"""
        try:
            # PyTorch cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"GPU cleanup warning: {e}")


def setup_robust_multiprocessing(
    max_memory_per_process: int = 8192,
    process_timeout: int = 3600,
    enable_monitoring: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    🛡️ Setup robust multiprocessing environment
    
    Konfiguriert das multiprocessing environment für optimale Performance
    und Stabilität.
    
    Parameters
    ----------
    max_memory_per_process : int
        Maximum memory per process in MB
    process_timeout : int
        Process timeout in seconds
    enable_monitoring : bool
        Enable process monitoring
    **kwargs
        Additional configuration options
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    
    logger.info("🛡️ Setting up robust multiprocessing environment")
    
    try:
        # Update global configuration
        ROBUST_MP_CONFIG.update({
            'max_memory_per_process': max_memory_per_process * 1024 * 1024,  # Convert to bytes
            'process_timeout': process_timeout,
            'enable_memory_monitoring': enable_monitoring and PSUTIL_AVAILABLE,
            **kwargs
        })
        
        # Set multiprocessing start method
        '''if hasattr(mp, 'set_start_method'):
            try:
                # Use spawn for better isolation
                mp.set_start_method('spawn', force=True)
                logger.info("✅ Set multiprocessing start method to 'spawn'")
            except RuntimeError as e:
                logger.info(f"Multiprocessing start method already set: {e}")
        '''
        mp.set_start_method('spawn', force=True) 
        # Configure process monitoring
        global _process_monitor
        _process_monitor = ProcessMonitor(max_memory_mb=max_memory_per_process)
        
        # Set environment variables for better multiprocessing
        os.environ['PYTHONHASHSEED'] = '0'  # Reproducible hashing
        import psutil
        available_cores = min(4, psutil.cpu_count())
        os.environ['OMP_NUM_THREADS'] = str(available_cores)
        os.environ['MKL_NUM_THREADS'] = str(available_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(available_cores)
        print(f"🚀 Multi-Threading aktiviert: {available_cores} Threads pro Prozess")
        #os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # GPU environment
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        logger.info("✅ Robust multiprocessing setup complete")
        
        return ROBUST_MP_CONFIG.copy()
        
    except Exception as e:
        logger.error(f"❌ Failed to setup robust multiprocessing: {e}")
        raise


@contextmanager
def robust_training_context(
    max_workers: Optional[int] = None,
    max_memory_per_process: int = 8192,
    process_timeout: int = 3600,
    enable_monitoring: bool = True,
    **kwargs
):
    """
    🎯 Robust training context manager
    
    Context manager für robustes multiprocessing training.
    Automatisches setup und cleanup.
    
    Parameters
    ----------
    max_workers : int, optional
        Maximum number of workers
    max_memory_per_process : int
        Maximum memory per process in MB
    process_timeout : int
        Process timeout in seconds
    enable_monitoring : bool
        Enable process monitoring
    **kwargs
        Additional arguments
        
    Yields
    ------
    RobustProcessPool
        Configured process pool
        
    Example
    -------
    >>> with robust_training_context(max_workers=4) as pool:
    ...     futures = [pool.submit(train_model, data) for data in datasets]
    ...     results = [f.result() for f in futures]
    """
    
    logger.info("🎯 Entering robust training context")
    
    # Setup environment
    config = setup_robust_multiprocessing(
        max_memory_per_process=max_memory_per_process,
        process_timeout=process_timeout,
        enable_monitoring=enable_monitoring,
        **kwargs
    )
    
    # Create robust process pool
    pool = RobustProcessPool(
        max_workers=max_workers,
        max_memory_per_process=max_memory_per_process,
        process_timeout=process_timeout,
        enable_monitoring=enable_monitoring,
        **kwargs
    )
    
    try:
        with pool:
            logger.info("✅ Robust training context ready")
            yield pool
            
    except Exception as e:
        logger.error(f"❌ Error in training context: {e}")
        raise
        
    finally:
        logger.info("🛑 Exiting robust training context")
        
        # Final cleanup
        try:
            gc.collect()
            
            # GPU cleanup
            if ROBUST_MP_CONFIG.get('enable_gpu_cleanup', True):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"⚠️ Final cleanup warning: {e}")


def robust_parallel_map(
    func: Callable,
    iterable,
    max_workers: Optional[int] = None,
    timeout: Optional[int] = None,
    show_progress: bool = True,
    **kwargs
) -> List[Any]:
    """
    🚀 Robust parallel map function
    
    High-level function für robustes parallel processing.
    
    Parameters
    ----------
    func : Callable
        Function to apply
    iterable : Iterable
        Data to process
    max_workers : int, optional
        Maximum number of workers
    timeout : int, optional
        Timeout in seconds
    show_progress : bool
        Show progress bar
    **kwargs
        Additional arguments for robust_training_context
        
    Returns
    -------
    List[Any]
        Results list
        
    Example
    -------
    >>> results = robust_parallel_map(
    ...     process_data,
    ...     data_files,
    ...     max_workers=4,
    ...     timeout=3600
    ... )
    """
    
    items = list(iterable)
    logger.info(f"🚀 Starting robust parallel map with {len(items)} items")
    
    with robust_training_context(max_workers=max_workers, **kwargs) as pool:
        
        if show_progress:
            try:
                from tqdm import tqdm
                #progress_bar = tqdm(total=len(items), desc="Processing")
                progress_bar = tqdm(disable=(mp.current_process().name != "MainProcess"), total=len(items), desc="Processing")
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None
        
        try:
            # Submit all tasks
            futures = [pool.submit(func, item) for item in items]
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                    
                    if progress_bar:
                        progress_bar.update(1)
                        
                except Exception as e:
                    logger.error(f"❌ Task failed: {e}")
                    results.append(None)
                    
                    if progress_bar:
                        progress_bar.update(1)
            
            if progress_bar:
                progress_bar.close()
            
            logger.info(f"✅ Robust parallel map completed: {len(results)} results")
            return results
            
        except Exception as e:
            if progress_bar:
                progress_bar.close()
            logger.error(f"❌ Robust parallel map failed: {e}")
            raise


def memory_efficient_batch_processor(
    func: Callable,
    data_generator,
    batch_size: int = 32,
    max_workers: Optional[int] = None,
    **kwargs
):
    """
    💾 Memory-efficient batch processor
    
    Prozessiert große datasets in batches um memory usage zu begrenzen.
    
    Parameters
    ----------
    func : Callable
        Processing function
    data_generator : Generator
        Data generator
    batch_size : int
        Batch size
    max_workers : int, optional
        Maximum workers
    **kwargs
        Additional arguments
        
    Yields
    ------
    Any
        Processed results
    """
    
    logger.info(f"💾 Starting memory-efficient batch processing (batch_size={batch_size})")
    
    with robust_training_context(max_workers=max_workers, **kwargs) as pool:
        
        batch = []
        batch_count = 0
        
        for item in data_generator:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch
                logger.debug(f"Processing batch {batch_count} with {len(batch)} items")
                
                futures = [pool.submit(func, item) for item in batch]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        yield result
                    except Exception as e:
                        logger.error(f"❌ Batch item failed: {e}")
                        yield None
                
                # Clear batch and force GC
                batch.clear()
                gc.collect()
                batch_count += 1
        
        # Process remaining items
        if batch:
            logger.debug(f"Processing final batch with {len(batch)} items")
            
            futures = [pool.submit(func, item) for item in batch]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    yield result
                except Exception as e:
                    logger.error(f"❌ Final batch item failed: {e}")
                    yield None
    
    logger.info("✅ Memory-efficient batch processing completed")


# Utility functions

def get_optimal_worker_count(
    cpu_intensive: bool = True,
    memory_per_worker_gb: float = 2.0,
    max_workers: Optional[int] = None
) -> int:
    """
    🎯 Get optimal worker count based on system resources
    
    Parameters
    ----------
    cpu_intensive : bool
        Whether tasks are CPU intensive
    memory_per_worker_gb : float
        Expected memory per worker in GB
    max_workers : int, optional
        Maximum workers override
        
    Returns
    -------
    int
        Optimal worker count
    """
    
    # Get system info
    cpu_count = 1 or 4
    
    if PSUTIL_AVAILABLE:
        memory_gb = psutil.virtual_memory().total / (1024**3)
    else:
        # Fallback estimate
        memory_gb = 8.0
    
    # Calculate based on CPU
    if cpu_intensive:
        cpu_workers = cpu_count
    else:
        cpu_workers = min(cpu_count * 2, 16)  # I/O bound can use more
    
    # Calculate based on memory
    memory_workers = int(memory_gb / memory_per_worker_gb)
    
    # Take minimum
    optimal = min(cpu_workers, memory_workers, max_workers or float('inf'))
    optimal = max(1, optimal)  # At least 1 worker
    
    logger.info(f"🎯 Optimal worker count: {optimal} (CPU: {cpu_workers}, Memory: {memory_workers})")
    
    return optimal


def cleanup_multiprocessing_resources():
    """
    🧹 Cleanup multiprocessing resources
    
    Vollständige cleanup aller multiprocessing resources.
    Sollte am Ende des Programms aufgerufen werden.
    """
    
    logger.info("🧹 Cleaning up multiprocessing resources")
    
    try:
        # Stop process monitoring
        _process_monitor.stop_monitoring()
        
        # Cleanup all registered processes
        for pid in list(_process_monitor.processes.keys()):
            _process_monitor.cleanup_process(pid, force=False)
        
        # Force garbage collection
        gc.collect()
        
        # GPU cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception:
            pass
        
        logger.info("✅ Multiprocessing resources cleaned up")
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")