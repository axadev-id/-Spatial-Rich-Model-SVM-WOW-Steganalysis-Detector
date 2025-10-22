"""
Utility Functions for Steganalysis Pipeline
===========================================

This module provides various utility functions for the steganalysis pipeline
including timing, logging, visualization, and performance monitoring.

Key features:
- Performance timing and monitoring
- GPU memory management
- Progress tracking and logging
- Visualization utilities
- Configuration management
"""

import time
import logging
import functools
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
import psutil
import os
from datetime import datetime
import pickle

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class PerformanceTimer:
    """
    Context manager and decorator for timing operations.
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.name} started...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.logger.info(f"{self.name} completed in {self.elapsed_time:.2f} seconds")
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(f"{func.__name__}", self.logger):
                return func(*args, **kwargs)
        return wrapper


class SystemMonitor:
    """
    Monitor system resources during execution.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cpu_percent = []
        self.memory_percent = []
        self.gpu_memory_used = []
        self.gpu_utilization = []
        self.timestamps = []
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        # GPU information
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    info.update({
                        'gpu_name': gpu.name,
                        'gpu_memory_total': gpu.memoryTotal,
                        'gpu_memory_used': gpu.memoryUsed,
                        'gpu_memory_free': gpu.memoryFree,
                        'gpu_utilization': gpu.load * 100,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception as e:
                self.logger.warning(f"Could not get GPU info: {str(e)}")
        
        # CuPy memory info
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                info.update({
                    'cupy_memory_used': mempool.used_bytes(),
                    'cupy_memory_total': mempool.total_bytes()
                })
            except Exception as e:
                self.logger.warning(f"Could not get CuPy memory info: {str(e)}")
        
        return info
    
    def log_system_status(self):
        """Log current system status."""
        info = self.get_system_info()
        
        self.logger.info("=== System Status ===")
        self.logger.info(f"CPU Usage: {info.get('cpu_percent', 0):.1f}%")
        self.logger.info(f"Memory Usage: {info.get('memory_percent', 0):.1f}%")
        
        if 'gpu_utilization' in info:
            self.logger.info(f"GPU Usage: {info['gpu_utilization']:.1f}%")
            self.logger.info(f"GPU Memory: {info['gpu_memory_used']:.0f}/{info['gpu_memory_total']:.0f} MB")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring."""
        import threading
        
        def monitor():
            while self.monitoring:
                info = self.get_system_info()
                self.cpu_percent.append(info.get('cpu_percent', 0))
                self.memory_percent.append(info.get('memory_percent', 0))
                self.gpu_memory_used.append(info.get('gpu_memory_used', 0))
                self.gpu_utilization.append(info.get('gpu_utilization', 0))
                self.timestamps.append(time.time())
                time.sleep(interval)
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if hasattr(self, 'monitoring'):
            self.monitoring = False
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join()
    
    def plot_monitoring_results(self, save_path: Optional[str] = None):
        """Plot monitoring results."""
        if not self.timestamps:
            self.logger.warning("No monitoring data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert timestamps to relative time
        start_time = self.timestamps[0]
        relative_times = [(t - start_time) / 60 for t in self.timestamps]  # Minutes
        
        # CPU usage
        axes[0, 0].plot(relative_times, self.cpu_percent, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage
        axes[0, 1].plot(relative_times, self.memory_percent, 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU utilization
        if any(u > 0 for u in self.gpu_utilization):
            axes[1, 0].plot(relative_times, self.gpu_utilization, 'r-', linewidth=2)
            axes[1, 0].set_title('GPU Utilization Over Time')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('GPU Utilization (%)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No GPU Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GPU Utilization Over Time')
        
        # GPU memory
        if any(m > 0 for m in self.gpu_memory_used):
            axes[1, 1].plot(relative_times, self.gpu_memory_used, 'm-', linewidth=2)
            axes[1, 1].set_title('GPU Memory Usage Over Time')
            axes[1, 1].set_xlabel('Time (minutes)')
            axes[1, 1].set_ylabel('GPU Memory (MB)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No GPU Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('GPU Memory Usage Over Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Monitoring plot saved to {save_path}")
        
        plt.show()


class ConfigManager:
    """
    Configuration management for the steganalysis pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                elif config_path.suffix.lower() in ['.yml', '.yaml']:
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
    
    def save_config(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                elif config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary."""
        def update_nested(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested(self.config, updates)


class ResultsLogger:
    """
    Logger for experiment results and metrics.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.results = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'metrics': {},
            'timing': {},
            'system_info': {}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def set_config(self, config: Dict[str, Any]):
        """Set experiment configuration."""
        self.results['config'] = config
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """Log a metric value."""
        if name not in self.results['metrics']:
            self.results['metrics'][name] = []
        
        entry = {'value': value, 'timestamp': datetime.now().isoformat()}
        if step is not None:
            entry['step'] = step
        
        self.results['metrics'][name].append(entry)
    
    def log_timing(self, name: str, duration: float):
        """Log timing information."""
        self.results['timing'][name] = duration
    
    def log_system_info(self, system_monitor: SystemMonitor):
        """Log system information."""
        self.results['system_info'] = system_monitor.get_system_info()
    
    def save_results(self):
        """Save results to file."""
        self.results['end_time'] = datetime.now().isoformat()
        
        # Save as JSON
        results_file = self.log_dir / f"{self.experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as pickle for complex objects
        pickle_file = self.log_dir / f"{self.experiment_name}_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def load_results(self, experiment_name: str):
        """Load results from file."""
        results_file = self.log_dir / f"{experiment_name}_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            self.logger.info(f"Results loaded from {results_file}")
        else:
            self.logger.warning(f"Results file not found: {results_file}")


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Log file path
        log_format: Log format string
        
    Returns:
        Configured logger
    """
    
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_directory(base_dir: str, experiment_name: str) -> Path:
    """
    Create directory structure for experiment.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to experiment directory
    """
    
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    return exp_dir


def save_array_to_disk(array: np.ndarray, 
                      filepath: str, 
                      compress: bool = True) -> None:
    """
    Save numpy array to disk with optional compression.
    
    Args:
        array: Numpy array to save
        filepath: Path to save file
        compress: Whether to use compression
    """
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        np.savez_compressed(filepath.with_suffix('.npz'), data=array)
    else:
        np.save(filepath.with_suffix('.npy'), array)


def load_array_from_disk(filepath: str) -> np.ndarray:
    """
    Load numpy array from disk.
    
    Args:
        filepath: Path to array file
        
    Returns:
        Loaded numpy array
    """
    
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        return np.load(filepath)['data']
    else:
        return np.load(filepath)


def plot_training_history(metrics: Dict[str, List], 
                         save_path: Optional[str] = None):
    """
    Plot training history metrics.
    
    Args:
        metrics: Dictionary of metric lists
        save_path: Path to save plot
    """
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        if isinstance(values[0], dict) and 'value' in values[0]:
            y_values = [v['value'] for v in values]
            x_values = range(len(y_values))
        else:
            y_values = values
            x_values = range(len(values))
        
        axes[i].plot(x_values, y_values, 'b-', linewidth=2)
        axes[i].set_title(f'{metric_name.title()} Over Time')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel(metric_name.title())
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Test performance timer
    with PerformanceTimer("Test Operation", logger):
        time.sleep(1)
    
    # Test system monitor
    monitor = SystemMonitor()
    monitor.log_system_status()
    
    # Test config manager
    config = ConfigManager()
    config.set('model.kernel', 'rbf')
    config.set('model.C', 1.0)
    print(f"Model config: {config.get('model')}")
    
    # Test results logger
    results = ResultsLogger("test_logs", "test_experiment")
    results.log_metric("accuracy", 0.95)
    results.log_timing("training", 120.5)
    results.save_results()
    
    print("Utils module test completed successfully!")