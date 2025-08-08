"""
Model configuration and management for Phi-3 integration
"""
import os
import psutil
from pathlib import Path
from typing import Dict, Optional
import logging

from config import MODEL_CONFIG, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

class ModelConfig:
    """
    Manages model configuration and system optimization for CPU inference
    """
    
    def __init__(self):
        self.config = MODEL_CONFIG.copy()
        self.system_info = self._get_system_info()
        self._optimize_for_system()
    
    def _get_system_info(self) -> Dict:
        """
        Get system information for optimization
        
        Returns:
            Dictionary with system information
        """
        return {
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'platform': os.name
        }
    
    def _optimize_for_system(self):
        """
        Optimize model configuration based on system capabilities
        """
        # Optimize thread count based on CPU cores
        cpu_cores = self.system_info['cpu_count']
        if cpu_cores:
            # Use 75% of available cores, leaving some for system processes
            optimal_threads = max(1, int(cpu_cores * 0.75))
            self.config['n_threads'] = min(optimal_threads, 16)  # Cap at 16 threads
            logger.info(f"Optimized thread count: {self.config['n_threads']} (CPU cores: {cpu_cores})")
        
        # Optimize context window based on available memory
        available_memory = self.system_info['available_memory_gb']
        if available_memory < 8:
            # Reduce context window for low memory systems
            self.config['n_ctx'] = 2048
            logger.warning(f"Reduced context window to {self.config['n_ctx']} due to low memory ({available_memory:.1f}GB)")
        elif available_memory > 12:
            # Can use full context window
            self.config['n_ctx'] = 4096
            logger.info(f"Using full context window: {self.config['n_ctx']}")
        
        # Ensure GPU layers is 0 for CPU-only inference
        self.config['n_gpu_layers'] = 0
        
        # Optimize batch size based on memory
        if available_memory < 8:
            self.config['n_batch'] = 128
        else:
            self.config['n_batch'] = 512
    
    def get_model_config(self) -> Dict:
        """
        Get optimized model configuration
        
        Returns:
            Dictionary with model configuration
        """
        return self.config.copy()
    
    def validate_model_path(self, model_path: Optional[str] = None) -> str:
        """
        Validate and return model path
        
        Args:
            model_path: Optional custom model path
            
        Returns:
            Validated model path
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = model_path or self.config['model_path']
        
        # Check if path is absolute or relative
        if not os.path.isabs(path):
            # Try different common locations
            possible_paths = [
                path,  # Current directory
                os.path.join(os.getcwd(), path),
                os.path.join(os.path.expanduser('~'), 'models', path),
                os.path.join(os.path.expanduser('~'), 'Downloads', path),
            ]
            
            for possible_path in possible_paths:
                if os.path.exists(possible_path):
                    logger.info(f"Found model at: {possible_path}")
                    return possible_path
            
            # If not found, provide helpful error message
            raise FileNotFoundError(
                f"Model file '{path}' not found. Please ensure the Phi-3-mini-4k-instruct-Q4_K_M.gguf file is in one of these locations:\n"
                f"- Current directory: {os.getcwd()}\n"
                f"- Home/models: {os.path.join(os.path.expanduser('~'), 'models')}\n"
                f"- Home/Downloads: {os.path.join(os.path.expanduser('~'), 'Downloads')}\n"
                f"You can download it from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
            )
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        return path
    
    def get_memory_usage_estimate(self) -> Dict:
        """
        Estimate memory usage for the model
        
        Returns:
            Dictionary with memory usage estimates
        """
        # Rough estimates for Q4_K_M quantization
        model_size_gb = 2.4  # Approximate size for Phi-3-mini Q4_K_M
        context_memory_gb = (self.config['n_ctx'] * 4) / (1024**3)  # Rough estimate
        
        total_estimate = model_size_gb + context_memory_gb + 1.0  # +1GB for overhead
        
        return {
            'model_size_gb': model_size_gb,
            'context_memory_gb': context_memory_gb,
            'total_estimate_gb': total_estimate,
            'available_memory_gb': self.system_info['available_memory_gb'],
            'memory_sufficient': total_estimate < self.system_info['available_memory_gb'] * 0.8
        }
    
    def get_performance_tips(self) -> List[str]:
        """
        Get performance optimization tips based on system
        
        Returns:
            List of performance tips
        """
        tips = []
        
        memory_info = self.get_memory_usage_estimate()
        
        if not memory_info['memory_sufficient']:
            tips.append("Consider closing other applications to free up memory")
            tips.append("Reduce context window (n_ctx) if experiencing memory issues")
        
        if self.system_info['cpu_count'] and self.system_info['cpu_count'] < 8:
            tips.append("Consider using a smaller model for better performance on this CPU")
        
        if self.system_info['available_memory_gb'] < 8:
            tips.append("System has limited RAM - expect slower inference times")
        
        tips.extend([
            "Ensure no other CPU-intensive applications are running",
            "Consider using SSD storage for faster model loading",
            "Monitor CPU temperature during extended use"
        ])
        
        return tips

