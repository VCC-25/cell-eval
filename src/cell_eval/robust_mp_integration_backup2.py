"""
Robust Multiprocessing Integration for Cell Evaluation

This module provides enhanced multiprocessing capabilities with automatic
error recovery, performance monitoring, and intelligent resource management.

Author: Enhanced Multiprocessing Integration
Version: 1.0.0
"""

import os
import sys
import time
import signal
import traceback
import threading
import queue
import warnings
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor

from multiprocessing import Manager, Queue, Event, Value
import multiprocessing as mp

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# === PDEX SPEED OPTIMIZATION START ===
import os
import multiprocessing as mp
import psutil

# Intelligente Ressourcen-Erkennung
def get_optimal_workers():
    cpu_count = mp.cpu_count()
    if HAS_PSUTIL:
        memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if memory_gb > 16:  # Viel RAM
            return min(cpu_count, 8)
        elif memory_gb > 8:  # Mittlerer RAM
            return min(cpu_count, 4)
        else:  # Wenig RAM
            return min(cpu_count, 2)
    else:
        return min(cpu_count, 4)  # Conservative default

OPTIMAL_WORKERS = get_optimal_workers()
OPTIMAL_THREADS = max(1, OPTIMAL_WORKERS // 2)

# Multi-Threading Environment
os.environ['OMP_NUM_THREADS'] = str(OPTIMAL_THREADS)
os.environ['MKL_NUM_THREADS'] = str(OPTIMAL_THREADS) 
os.environ['NUMEXPR_NUM_THREADS'] = str(OPTIMAL_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(OPTIMAL_THREADS)

print(f"🚀 Parallelisierung: {OPTIMAL_WORKERS} Workers, {OPTIMAL_THREADS} Threads/Worker")

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
class ProcessMetrics:
    """Metrics for individual process performance"""
    process_id: int
    start_time: float
    end_time: Optional[float] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    memory_peak: float = 0.0
    cpu_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get process duration"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Get success rate"""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0


class RobustWorkerProcess:
    """Enhanced worker process with monitoring and error recovery"""
    
    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue, 
                 error_queue: Queue, stop_event: Event, metrics_queue: Queue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.error_queue = error_queue
        self.stop_event = stop_event
        self.metrics_queue = metrics_queue
        
        # Process metrics
        self.metrics = ProcessMetrics(
            process_id=worker_id,
            start_time=time.time()
        )
        
        # Process monitoring
        self.process = None
        if HAS_PSUTIL:
            try:
                self.process = psutil.Process()
            except:
                pass
    
    def run(self, task_function: Callable):
        """Main worker process loop"""
        try:
            # Set up signal handlers
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            while not self.stop_event.is_set():
                try:
                    # Get task with timeout
                    task = self.task_queue.get(timeout=1.0)
                    if task is None:  # Poison pill
                        break
                    
                    # Process task
                    start_time = time.time()
                    try:
                        result = task_function(task)
                        self.result_queue.put((self.worker_id, task, result, None))
                        self.metrics.tasks_completed += 1
                    except Exception as e:
                        error_info = {
                            'worker_id': self.worker_id,
                            'task': task,
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'timestamp': time.time()
                        }
                        self.error_queue.put(error_info)
                        self.result_queue.put((self.worker_id, task, None, error_info))
                        self.metrics.tasks_failed += 1
                        self.metrics.errors.append(str(e))
                    
                    # Update metrics
                    if self.process:
                        try:
                            memory_info = self.process.memory_info()
                            self.metrics.memory_peak = max(
                                self.metrics.memory_peak, 
                                memory_info.rss / 1024 / 1024  # MB
                            )
                            self.metrics.cpu_time = self.process.cpu_times().user
                        except:
                            pass
                    
                    # Report metrics periodically
                    if (self.metrics.tasks_completed + self.metrics.tasks_failed) % 10 == 0:
                        self.metrics_queue.put(self.metrics)
                
                except queue.Empty:
                    continue
                except Exception as e:
                    self.error_queue.put({
                        'worker_id': self.worker_id,
                        'error': f"Worker loop error: {e}",
                        'traceback': traceback.format_exc(),
                        'timestamp': time.time()
                    })
        
        finally:
            # Final metrics report
            self.metrics.end_time = time.time()
            self.metrics_queue.put(self.metrics)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        self.stop_event.set()


class AutoRobustProcessPool:
    """
    Automatic robust process pool with intelligent error recovery,
    performance monitoring, and adaptive configuration.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 workload_type: str = 'general',
                 enable_monitoring: bool = True,
                 auto_tune: bool = True,
                 error_threshold: float = 0.1,
                 restart_threshold: int = 5):
        """
        Initialize the robust process pool
        
        Args:
            max_workers: Maximum number of worker processes
            workload_type: Type of workload for optimization
            enable_monitoring: Enable performance monitoring
            auto_tune: Enable automatic tuning
            error_threshold: Error rate threshold for worker restart
            restart_threshold: Number of errors before restarting worker
        """
        self.max_workers = max_workers or OPTIMAL_WORKERS
        self.workload_type = workload_type
        self.enable_monitoring = enable_monitoring
        self.auto_tune = auto_tune
        self.error_threshold = error_threshold
        self.restart_threshold = restart_threshold
        
        # Process management
        self.workers: Dict[int, mp.Process] = {}
        self.worker_metrics: Dict[int, ProcessMetrics] = {}
        self.is_running = False
        
        # Queues and events
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        self.metrics_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        
        # Monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.error_monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
        
        # Auto-tuning
        self.performance_history: List[Dict[str, Any]] = []
        self.last_tune_time = 0
        self.tune_interval = 30.0  # seconds
    
    def start(self):
        """Start the process pool"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.stop_event.clear()
        
        # Start worker processes
        for worker_id in range(self.max_workers):
            self._start_worker(worker_id)
        
        # Start monitoring threads
        if self.enable_monitoring:
            self._start_monitoring()
        
        print(f"🚀 AutoRobustProcessPool started with {self.max_workers} workers")
    
    def _start_worker(self, worker_id: int):
        """Start a single worker process"""
        worker = RobustWorkerProcess(
            worker_id=worker_id,
            task_queue=self.task_queue,
            result_queue=self.result_queue,
            error_queue=self.error_queue,
            stop_event=self.stop_event,
            metrics_queue=self.metrics_queue
        )
        
        process = mp.Process(target=worker.run, args=(self._default_task_function,))
        process.start()
        self.workers[worker_id] = process
    
    def _default_task_function(self, task):
        """Default task function - should be overridden"""
        return task
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.error_monitor_thread = threading.Thread(target=self._error_monitor_loop, daemon=True)
        
        self.monitor_thread.start()
        self.error_monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitor worker performance"""
        while self.is_running:
            try:
                metrics = self.metrics_queue.get(timeout=1.0)
                self.worker_metrics[metrics.process_id] = metrics
                
                # Auto-tuning
                if self.auto_tune and time.time() - self.last_tune_time > self.tune_interval:
                    self._auto_tune()
                    self.last_tune_time = time.time()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Monitor error: {e}")
    
    def _error_monitor_loop(self):
        """Monitor and handle errors"""
        while self.is_running:
            try:
                error_info = self.error_queue.get(timeout=1.0)
                worker_id = error_info.get('worker_id')
                
                if worker_id is not None:
                    # Check if worker needs restart
                    metrics = self.worker_metrics.get(worker_id)
                    if metrics and len(metrics.errors) >= self.restart_threshold:
                        self._restart_worker(worker_id)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error monitor error: {e}")
    
    def _restart_worker(self, worker_id: int):
        """Restart a failed worker"""
        try:
            # Terminate old worker
            if worker_id in self.workers:
                self.workers[worker_id].terminate()
                self.workers[worker_id].join(timeout=5.0)
                del self.workers[worker_id]
            
            # Start new worker
            self._start_worker(worker_id)
            print(f"🔄 Worker {worker_id} restarted")
            
        except Exception as e:
            print(f"Failed to restart worker {worker_id}: {e}")
    
    def _auto_tune(self):
        """Automatically tune performance parameters"""
        if not self.worker_metrics:
            return
        
        # Calculate overall performance
        total_completed = sum(m.tasks_completed for m in self.worker_metrics.values())
        total_failed = sum(m.tasks_failed for m in self.worker_metrics.values())
        error_rate = total_failed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'workers': len(self.workers),
            'completed': total_completed,
            'failed': total_failed,
            'error_rate': error_rate
        })
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
    
    def submit_tasks(self, tasks: List[Any], task_function: Callable):
        """Submit tasks to the pool"""
        self.total_tasks += len(tasks)
        
        # Store task function
        self._current_task_function = task_function
        
        # Submit tasks
        for task in tasks:
            self.task_queue.put(task)
    
    def get_results(self, timeout: Optional[float] = None) -> List[Any]:
        """Get all results"""
        results = []
        start_time = time.time()
        
        while len(results) < self.total_tasks:
            try:
                result_timeout = 1.0 if timeout is None else min(1.0, timeout - (time.time() - start_time))
                if result_timeout <= 0:
                    break
                    
                worker_id, task, result, error = self.result_queue.get(timeout=result_timeout)
                
                if error is None:
                    results.append(result)
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1
                    
            except queue.Empty:
                if timeout and time.time() - start_time > timeout:
                    break
                continue
        
        return results
    
    def shutdown(self, wait: bool = True):
        """Shutdown the process pool"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Send poison pills
        for _ in range(self.max_workers):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        if wait:
            for worker in self.workers.values():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
        
        # Clean up
        self.workers.clear()
        self.manager.shutdown()
        
        print("🛑 AutoRobustProcessPool shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'workers': len(self.workers),
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0,
            'runtime': runtime,
            'tasks_per_second': self.completed_tasks / runtime if runtime > 0 else 0,
            'worker_metrics': dict(self.worker_metrics)
        }


# === HAUPTFUNKTIONEN FÜR IMPORT ===

def auto_pool(max_workers: Optional[int] = None, 
              workload_type: str = 'general',
              **kwargs) -> AutoRobustProcessPool:
    """
    Create an automatic robust process pool
    
    Args:
        max_workers: Maximum number of workers (auto-detected if None)
        workload_type: Type of workload ('general', 'cpu_intensive', 'memory_intensive')
        **kwargs: Additional arguments for AutoRobustProcessPool
    
    Returns:
        AutoRobustProcessPool instance
    """
    if max_workers is None:
        if workload_type == 'cpu_intensive':
            max_workers = OPTIMAL_WORKERS
        elif workload_type == 'memory_intensive':
            max_workers = max(1, OPTIMAL_WORKERS // 2)
        else:
            max_workers = min(OPTIMAL_WORKERS, 4)
    
    return AutoRobustProcessPool(
        max_workers=max_workers,
        workload_type=workload_type,
        **kwargs
    )


def process_batch(tasks: List[Any], 
                  task_function: Callable,
                  max_workers: Optional[int] = None,
                  timeout: Optional[float] = None,
                  show_progress: bool = True) -> List[Any]:
    """
    Process a batch of tasks with automatic pool management
    
    Args:
        tasks: List of tasks to process
        task_function: Function to process each task
        max_workers: Number of workers (auto-detected if None)
        timeout: Timeout in seconds
        show_progress: Show progress bar
    
    Returns:
        List of results
    """
    pool = auto_pool(max_workers=max_workers)
    
    try:
        pool.start()
        pool.submit_tasks(tasks, task_function)
        
        if show_progress:
            try:
                import tqdm
                with tqdm.tqdm(total=len(tasks), desc="Processing") as pbar:
                    results = []
                    while len(results) < len(tasks):
                        batch_results = pool.get_results(timeout=1.0)
                        new_results = len(batch_results) - len(results)
                        if new_results > 0:
                            pbar.update(new_results)
                        results = batch_results
                    return results
            except ImportError:
                return pool.get_results(timeout=timeout)
        else:
            return pool.get_results(timeout=timeout)
    
    finally:
        pool.shutdown()


# Convenience functions
def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization"""
    info = {
        'cpu_count': mp.cpu_count(),
        'optimal_workers': OPTIMAL_WORKERS,
        'optimal_threads': OPTIMAL_THREADS,
        'platform': platform.system(),
        'mp_method': mp.get_start_method(),
        'has_psutil': HAS_PSUTIL
    }
    
    if HAS_PSUTIL:
        info.update({
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
        })
    
    return info


# Export main functions
__all__ = [
    'auto_pool',
    'process_batch', 
    'AutoRobustProcessPool',
    'ProcessMetrics',
    'get_system_info',
    'OPTIMAL_WORKERS',
    'OPTIMAL_THREADS'
]
