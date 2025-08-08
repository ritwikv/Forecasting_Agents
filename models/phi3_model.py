"""
Phi-3 model integration using llama-cpp-python for CPU inference
"""
import logging
from typing import Dict, List, Optional, Any
import time
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class Phi3Model:
    """
    Phi-3 model wrapper for CPU inference using llama-cpp-python
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Phi-3 model
        
        Args:
            model_path: Optional path to GGUF model file
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. Please install it with:\n"
                "pip install llama-cpp-python"
            )
        
        self.config_manager = ModelConfig()
        self.model_path = self.config_manager.validate_model_path(model_path)
        self.config = self.config_manager.get_model_config()
        self.model = None
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times = []
        self.token_counts = []
    
    def load_model(self) -> bool:
        """
        Load the Phi-3 model
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading Phi-3 model from: {self.model_path}")
            
            # Check memory requirements
            memory_info = self.config_manager.get_memory_usage_estimate()
            if not memory_info['memory_sufficient']:
                logger.warning(
                    f"Estimated memory usage ({memory_info['total_estimate_gb']:.1f}GB) "
                    f"may exceed available memory ({memory_info['available_memory_gb']:.1f}GB)"
                )
            
            start_time = time.time()
            
            # Initialize model with optimized settings
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.config['n_ctx'],
                n_threads=self.config['n_threads'],
                n_gpu_layers=self.config['n_gpu_layers'],
                verbose=self.config['verbose']
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def generate(self, 
                prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None,
                stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate text using the Phi-3 model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            
        Returns:
            Dictionary with generation results
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Model not loaded and failed to load")
        
        # Use config defaults if parameters not provided
        max_tokens = max_tokens or self.config['max_tokens']
        temperature = temperature or self.config['temperature']
        top_p = top_p or self.config['top_p']
        top_k = top_k or self.config['top_k']
        
        try:
            start_time = time.time()
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop or [],
                echo=False
            )
            
            inference_time = time.time() - start_time
            
            # Extract response text
            generated_text = response['choices'][0]['text']
            
            # Track performance
            self.inference_times.append(inference_time)
            self.token_counts.append(len(generated_text.split()))
            
            result = {
                'text': generated_text.strip(),
                'inference_time': inference_time,
                'tokens_generated': len(generated_text.split()),
                'tokens_per_second': len(generated_text.split()) / inference_time if inference_time > 0 else 0,
                'prompt_tokens': len(prompt.split()),
                'total_tokens': len(prompt.split()) + len(generated_text.split())
            }
            
            logger.debug(f"Generated {result['tokens_generated']} tokens in {inference_time:.2f}s "
                        f"({result['tokens_per_second']:.1f} tokens/s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Chat interface for conversational interactions
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with chat response
        """
        # Convert messages to Phi-3 chat format
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        result = self.generate(prompt, **kwargs)
        
        # Format as chat response
        result['message'] = {
            'role': 'assistant',
            'content': result['text']
        }
        
        return result
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into Phi-3 chat prompt format
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        formatted_messages = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_messages.append(f"<|system|>\n{content}<|end|>")
            elif role == 'user':
                formatted_messages.append(f"<|user|>\n{content}<|end|>")
            elif role == 'assistant':
                formatted_messages.append(f"<|assistant|>\n{content}<|end|>")
        
        # Add assistant prompt for response
        formatted_messages.append("<|assistant|>\n")
        
        return "\n".join(formatted_messages)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {'message': 'No inference data available'}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        avg_tokens = sum(self.token_counts) / len(self.token_counts)
        avg_tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            'total_inferences': len(self.inference_times),
            'average_inference_time': avg_time,
            'average_tokens_generated': avg_tokens,
            'average_tokens_per_second': avg_tokens_per_sec,
            'fastest_inference': min(self.inference_times),
            'slowest_inference': max(self.inference_times)
        }
    
    def clear_performance_stats(self):
        """Clear performance tracking data"""
        self.inference_times.clear()
        self.token_counts.clear()
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded successfully")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.unload_model()

