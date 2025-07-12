"""
Configuration management for the headshot generator.
"""

import os
from typing import Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Configuration class for the headshot generator."""
    
    # AI Model Settings
    DEVICE: str = os.getenv('DEVICE', 'auto')
    STABLE_DIFFUSION_MODEL: str = os.getenv('STABLE_DIFFUSION_MODEL', 'runwayml/stable-diffusion-v1-5')
    ENABLE_XFORMERS: bool = os.getenv('ENABLE_XFORMERS', 'true').lower() == 'true'
    LOW_MEMORY_MODE: bool = os.getenv('LOW_MEMORY_MODE', 'false').lower() == 'true'
    
    # Image Processing Settings
    DEFAULT_OUTPUT_SIZE: int = int(os.getenv('DEFAULT_OUTPUT_SIZE', '512'))
    MAX_IMAGE_SIZE: int = int(os.getenv('MAX_IMAGE_SIZE', '2048'))
    JPEG_QUALITY: int = int(os.getenv('JPEG_QUALITY', '95'))
    PNG_COMPRESSION: int = int(os.getenv('PNG_COMPRESSION', '6'))
    
    # Background Removal Settings
    BACKGROUND_MODEL: str = os.getenv('BACKGROUND_MODEL', 'u2net')
    
    # Face Detection Settings
    MIN_DETECTION_CONFIDENCE: float = float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.5'))
    FACE_MODEL_SELECTION: int = int(os.getenv('FACE_MODEL_SELECTION', '1'))
    
    # Web Interface Settings
    STREAMLIT_PORT: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    STREAMLIT_HOST: str = os.getenv('STREAMLIT_HOST', 'localhost')
    MAX_UPLOAD_SIZE: int = int(os.getenv('MAX_UPLOAD_SIZE', '10'))  # MB
    ENABLE_BATCH_PROCESSING: bool = os.getenv('ENABLE_BATCH_PROCESSING', 'true').lower() == 'true'
    
    # API Settings
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    ENABLE_CORS: bool = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'headshot_generator.log')
    ENABLE_FILE_LOGGING: bool = os.getenv('ENABLE_FILE_LOGGING', 'false').lower() == 'true'
    
    # Performance Settings
    TORCH_COMPILE: bool = os.getenv('TORCH_COMPILE', 'false').lower() == 'true'
    USE_HALF_PRECISION: bool = os.getenv('USE_HALF_PRECISION', 'true').lower() == 'true'
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '1'))
    NUM_WORKERS: int = int(os.getenv('NUM_WORKERS', '4'))
    
    # Storage Settings
    TEMP_DIR: Path = Path(os.getenv('TEMP_DIR', './temp'))
    OUTPUT_DIR: Path = Path(os.getenv('OUTPUT_DIR', './outputs'))
    CACHE_DIR: Path = Path(os.getenv('CACHE_DIR', './cache'))
    ENABLE_CACHING: bool = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
    CACHE_DURATION: int = int(os.getenv('CACHE_DURATION', '3600'))  # seconds
    
    # Safety Settings
    ENABLE_SAFETY_CHECKER: bool = os.getenv('ENABLE_SAFETY_CHECKER', 'false').lower() == 'true'
    NSFW_FILTER: bool = os.getenv('NSFW_FILTER', 'false').lower() == 'true'
    CONTENT_FILTER: bool = os.getenv('CONTENT_FILTER', 'false').lower() == 'true'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.TEMP_DIR, cls.OUTPUT_DIR, cls.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device_info(cls) -> str:
        """Get information about the selected device."""
        import torch
        
        if cls.DEVICE == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                device_info = f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name()}"
            elif torch.backends.mps.is_available():
                device = 'mps'
                device_info = "Apple Metal Performance Shaders"
            else:
                device = 'cpu'
                device_info = "CPU"
        else:
            device = cls.DEVICE
            if device == 'cuda' and torch.cuda.is_available():
                device_info = f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name()}"
            elif device == 'mps' and torch.backends.mps.is_available():
                device_info = "Apple Metal Performance Shaders"
            else:
                device_info = "CPU"
        
        return f"{device.upper()}: {device_info}"
    
    @classmethod
    def validate_config(cls) -> list:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check if directories are writable
        for name, directory in [
            ('TEMP_DIR', cls.TEMP_DIR),
            ('OUTPUT_DIR', cls.OUTPUT_DIR),
            ('CACHE_DIR', cls.CACHE_DIR)
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                test_file = directory / '.write_test'
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                issues.append(f"{name} ({directory}) is not writable: {e}")
        
        # Validate numeric ranges
        if cls.MIN_DETECTION_CONFIDENCE < 0 or cls.MIN_DETECTION_CONFIDENCE > 1:
            issues.append("MIN_DETECTION_CONFIDENCE must be between 0 and 1")
        
        if cls.MAX_IMAGE_SIZE < 256:
            issues.append("MAX_IMAGE_SIZE must be at least 256")
        
        if cls.DEFAULT_OUTPUT_SIZE < 64 or cls.DEFAULT_OUTPUT_SIZE > cls.MAX_IMAGE_SIZE:
            issues.append(f"DEFAULT_OUTPUT_SIZE must be between 64 and {cls.MAX_IMAGE_SIZE}")
        
        # Check device availability
        try:
            import torch
            if cls.DEVICE == 'cuda' and not torch.cuda.is_available():
                issues.append("CUDA device specified but not available")
            elif cls.DEVICE == 'mps' and not torch.backends.mps.is_available():
                issues.append("MPS device specified but not available")
        except ImportError:
            issues.append("PyTorch not installed - required for AI functionality")
        
        return issues

# Global configuration instance
config = Config()

# Ensure directories exist
config.create_directories()
