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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Event, Value
import multiprocessing as mp

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# === PDEX SPEED OPTIMIZATION START ===
#import os
#import multiprocessing as mp
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
        self.max_workers = max_workers or 1
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
        if worker_id in self.workers and self.workers[worker_id].is_alive():
            return
        
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
        print(f"✅ Started worker {worker_id} (PID: {process.pid})")
    
    def _default_task_function(self, task):
        """Default task function - can be overridden"""
        if callable(task):
            return task()
        elif isinstance(task, (list, tuple)) and len(task) >= 2:
            func, args = task[0], task[1:]
            return func(*args)
        else:
            return task
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # Metrics monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Error monitoring
        self.error_monitor_thread = threading.Thread(
            target=self._error_monitor_loop,
            daemon=True
        )
        self.error_monitor_thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics
                while not self.metrics_queue.empty():
                    try:
                        metrics = self.metrics_queue.get_nowait()
                        self.worker_metrics[metrics.process_id] = metrics
                    except queue.Empty:
                        break
                
                # Check for dead workers
                self._check_worker_health()
                
                # Auto-tune if enabled
                if (self.auto_tune and 
                    time.time() - self.last_tune_time > self.tune_interval):
                    self._auto_tune()
                    self.last_tune_time = time.time()
                
                time.sleep(1.0)
                
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
    
    def _error_monitor_loop(self):
        """Error monitoring loop"""
        while self.is_running:
            try:
                while not self.error_queue.empty():
                    try:
                        error_info = self.error_queue.get_nowait()
                        self._handle_worker_error(error_info)
                    except queue.Empty:
                        break
                
                time.sleep(0.5)
                
            except Exception as e:
                warnings.warn(f"Error monitoring error: {e}")
    
    def _handle_worker_error(self, error_info: Dict[str, Any]):
        """Handle worker errors"""
        worker_id = error_info.get('worker_id')
        error_msg = error_info.get('error', 'Unknown error')
        
        print(f"⚠️  Worker {worker_id} error: {error_msg}")
        
        # Check if worker needs restart
        if worker_id in self.worker_metrics:
            metrics = self.worker_metrics[worker_id]
            error_rate = metrics.tasks_failed / max(1, metrics.tasks_completed + metrics.tasks_failed)
            
            if (error_rate > self.error_threshold or 
                len(metrics.errors) >= self.restart_threshold):
                print(f"🔄 Restarting worker {worker_id} due to high error rate")
                self._restart_worker(worker_id)
    
    def _restart_worker(self, worker_id: int):
        """Restart a specific worker"""
        # Terminate old worker
        if worker_id in self.workers:
            try:
                self.workers[worker_id].terminate()
                self.workers[worker_id].join(timeout=5)
            except:
                pass
        
        # Start new worker
        self._start_worker(worker_id)
    
    def _check_worker_health(self):
        """Check health of all workers"""
        for worker_id, process in list(self.workers.items()):
            if not process.is_alive():
                print(f"💀 Worker {worker_id} died, restarting...")
                self._restart_worker(worker_id)
    
    def _auto_tune(self):
        """Automatic performance tuning"""
        if not self.worker_metrics:
            return
        
        # Calculate current performance metrics
        total_completed = sum(m.tasks_completed for m in self.worker_metrics.values())
        total_failed = sum(m.tasks_failed for m in self.worker_metrics.values())
        
        if total_completed == 0:
            return
        
        current_performance = {
            'timestamp': time.time(),
            'workers': len(self.workers),
            'completed_tasks': total_completed,
            'failed_tasks': total_failed,
            'success_rate': total_completed / (total_completed + total_failed),
            'avg_memory': sum(m.memory_peak for m in self.worker_metrics.values()) / len(self.worker_metrics)
        }
        
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Simple auto-tuning logic
        if len(self.performance_history) >= 3:
            recent_success_rates = [p['success_rate'] for p in self.performance_history[-3:]]
            avg_success_rate = sum(recent_success_rates) / len(recent_success_rates)
            
            if avg_success_rate < 0.8 and len(self.workers) > 1:
                # Reduce workers if success rate is low
                self._adjust_worker_count(len(self.workers) - 1)
            elif avg_success_rate > 0.95 and len(self.workers) < 1:
                # Increase workers if success rate is high
                self._adjust_worker_count(len(self.workers) + 1)
    
    def _adjust_worker_count(self, new_count: int):
        """Adjust the number of workers"""
        current_count = len(self.workers)
        
        if new_count > current_count:
            # Add workers
            for worker_id in range(current_count, new_count):
                self._start_worker(worker_id)
            print(f"📈 Increased workers from {current_count} to {new_count}")
        
        elif new_count < current_count:
            # Remove workers
            workers_to_remove = list(self.workers.keys())[new_count:]
            for worker_id in workers_to_remove:
                try:
                    self.workers[worker_id].terminate()
                    self.workers[worker_id].join(timeout=5)
                    del self.workers[worker_id]
                except:
                    pass
            print(f"📉 Decreased workers from {current_count} to {new_count}")
    
    def map(self, func: Callable, iterable, timeout: Optional[float] = None) -> List[Any]:
        """
        Map function over iterable using the process pool
        
        Args:
            func: Function to apply
            iterable: Iterable of items to process
            timeout: Optional timeout for the entire operation
        
        Returns:
            List of results
        """
        if not self.is_running:
            self.start()
        
        items = list(iterable)
        self.total_tasks = len(items)
        
        # Submit all tasks
        for item in items:
            self.task_queue.put((func, item))
        
        # Collect results
        results = [None] * len(items)
        completed = 0
        start_time = time.time()
        
        while completed < len(items):
            try:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")
                
                # Get result
                worker_id, original_task, result, error = self.result_queue.get(timeout=1.0)
                
                # Find original index
                original_item = original_task[1] if isinstance(original_task, tuple) else original_task
                try:
                    index = items.index(original_item)
                    if error is None:
                        results[index] = result
                        self.completed_tasks += 1
                    else:
                        results[index] = error  # or raise exception
                        self.failed_tasks += 1
                    completed += 1
                except ValueError:
                    # Item not found in original list
                    pass
                
            except queue.Empty:
                continue
        
        return results
    
    def close(self):
        """Close the process pool"""
        if not self.is_running:
            return
        
        print("🛑 Shutting down AutoRobustProcessPool...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Send poison pills
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker_id, process in self.workers.items():
            try:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
            except:
                pass
        
        # Clean up
        self.workers.clear()
        self.worker_metrics.clear()
        
        print("✅ AutoRobustProcessPool shut down complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'is_running': self.is_running,
            'workers': len(self.workers),
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.completed_tasks / max(1, self.total_tasks),
            'runtime': runtime,
            'tasks_per_second': self.completed_tasks / max(1, runtime),
            'worker_metrics': {k: v.__dict__ for k, v in self.worker_metrics.items()},
            'performance_history': self.performance_history[-5:]  # Last 5 entries
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience functions
def robust_map(func: Callable, iterable, 
               max_workers: Optional[int] = None,
               workload_type: str = 'general',
               timeout: Optional[float] = None) -> List[Any]:
    """
    Robust parallel map with automatic error recovery
    
    Args:
        func: Function to apply
        iterable: Iterable of items
        max_workers: Maximum number of workers
        workload_type: Type of workload
        timeout: Operation timeout
    
    Returns:
        List of results
    """
    with AutoRobustProcessPool(
        max_workers=max_workers,
        workload_type=workload_type,
        enable_monitoring=True,
        auto_tune=True
    ) as pool:
        async_result = pool.map_async(func, iterable)
        try:
            return async_result.get(timeout=timeout)
        except mp.TimeoutError:
            print("⚠️ Timeout erreicht, breche ab...")
            pool.terminate()
            pool.join()
            raise TimeoutError("Operation timed out")
            #return pool.map(func, iterable, timeout)

'''
if __name__ == "__main__":
    # Demo
    def test_function(x):
        """Test function for demonstration"""
        import time
        import random
        
        # Simulate work
        time.sleep(random.uniform(0.1, 0.5))
        
        # Simulate occasional errors
        if random.random() < 0.1:
            raise ValueError(f"Simulated error for {x}")
        
        return x * x
    
    print("🧪 Testing AutoRobustProcessPool...")
    
    # Test with robust map
    items = list(range(20))
    results = robust_map(test_function, items, max_workers=4, timeout=30)
    
    print(f"✅ Processed {len(items)} items")
    print(f"📊 Results: {results[:5]}... (showing first 5)")
'''