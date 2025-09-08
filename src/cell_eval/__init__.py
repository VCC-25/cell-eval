from ._baseline import build_base_mean_adata
from ._evaluator import MetricsEvaluator
#from ._optimized_evaluator import OptimizedMetricsEvaluator
from ._pipeline import KNOWN_PROFILES, MetricPipeline
from ._score import score_agg_metrics
from ._types import (
    BulkArrays,
    CellArrays,
    DEComparison,
    DEResults,
    DESortBy,
    MetricType,
    PerturbationAnndataPair,
    initialize_de_comparison,
)
from .metrics import metrics_registry

__all__ = [
    # Evaluation
    "MetricsEvaluator",
    # Baseline
    "build_base_mean_adata",
    # Scoring
    "score_agg_metrics",
    # Types
    "DEComparison",
    "DEResults",
    "DESortBy",
    "MetricType",
    "PerturbationAnndataPair",
    "BulkArrays",
    "CellArrays",
    "initialize_de_comparison",
    # Pipeline
    "MetricPipeline",
    "KNOWN_PROFILES",
    # Global registry
    "metrics_registry",
]

# ================================================================
# ENHANCED MULTIPROCESSING INTEGRATION
# ================================================================

# Import new system adapter and integration modules
try:
    from .dynamic_system_adapter import (
        DynamicSystemAdapter,
        get_optimal_config,
        print_system_info,
        benchmark_system_performance
    )
    DYNAMIC_ADAPTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DynamicSystemAdapter not available: {e}")
    DYNAMIC_ADAPTER_AVAILABLE = False

try:
    from .robust_mp_integration import (
        AutoRobustProcessPool,
        auto_pool,
        quick_map,
        benchmark_configurations
    )
    ROBUST_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RobustMPIntegration not available: {e}")
    ROBUST_INTEGRATION_AVAILABLE = False

# Enhanced exports - add to your existing __all__ list
__all__ = [
    # ... your existing exports ...
    
    # New enhanced multiprocessing exports
    'DynamicSystemAdapter',
    'AutoRobustProcessPool', 
    'auto_pool',
    'quick_map',
    'get_optimal_config',
    'print_system_info',
    'benchmark_configurations',
    'benchmark_system_performance',
    'DYNAMIC_ADAPTER_AVAILABLE',
    'ROBUST_INTEGRATION_AVAILABLE'
]

# Convenience function for quick setup
def setup_enhanced_multiprocessing(workload_type="general", enable_monitoring=True):
    """
    Quick setup function for enhanced multiprocessing
    
    Args:
        workload_type (str): Type of workload ("general", "cpu_intensive", etc.)
        enable_monitoring (bool): Enable performance monitoring
    
    Returns:
        AutoRobustProcessPool: Configured process pool
    """
    if not ROBUST_INTEGRATION_AVAILABLE:
        raise ImportError("Enhanced multiprocessing not available. Install required dependencies.")
    
    return AutoRobustProcessPool(
        workload_type=workload_type,
        enable_monitoring=enable_monitoring,
        auto_tune=True
    )