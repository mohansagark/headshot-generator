"""
Utility functions for the headshot generator.
"""

import os
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_image_hash(image_path: Union[str, Path]) -> str:
    """
    Generate a hash for an image file for caching purposes.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        SHA-256 hash of the image file
    """
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a valid image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    Get information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                'file_size': os.path.getsize(image_path)
            }
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {e}")
        return {}

def resize_image(image: Image.Image, max_size: int, maintain_aspect: bool = True) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum size for the longest side
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    else:
        image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image

def create_thumbnail(image_path: Union[str, Path], output_path: Union[str, Path], size: int = 256):
    """
    Create a thumbnail of an image.
    
    Args:
        image_path: Path to the source image
        output_path: Path where thumbnail will be saved
        size: Thumbnail size (square)
    """
    try:
        with Image.open(image_path) as img:
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            img.save(output_path, quality=85, optimize=True)
    except Exception as e:
        logger.error(f"Failed to create thumbnail: {e}")

def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24):
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Directory containing temporary files
        max_age_hours: Maximum age in hours before deletion
    """
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in temp_path.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file_path}: {e}")

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_supported_formats() -> List[str]:
    """
    Get list of supported image formats.
    
    Returns:
        List of supported file extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

def is_supported_format(file_path: Union[str, Path]) -> bool:
    """
    Check if file format is supported.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if format is supported, False otherwise
    """
    extension = Path(file_path).suffix.lower()
    return extension in get_supported_formats()

def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    # Ensure it's not empty
    if not safe_name:
        safe_name = 'untitled'
    
    return safe_name

def create_unique_filename(directory: Union[str, Path], base_name: str, extension: str) -> str:
    """
    Create a unique filename in the given directory.
    
    Args:
        directory: Target directory
        base_name: Base name for the file
        extension: File extension (with or without dot)
        
    Returns:
        Unique filename
    """
    directory = Path(directory)
    extension = extension if extension.startswith('.') else f'.{extension}'
    
    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}_{counter}{extension}"
        
        if not (directory / filename).exists():
            return filename
        
        counter += 1

def estimate_processing_time(image_size: Tuple[int, int], has_ai_generation: bool = False) -> int:
    """
    Estimate processing time based on image size and operations.
    
    Args:
        image_size: (width, height) of the image
        has_ai_generation: Whether AI generation is involved
        
    Returns:
        Estimated time in seconds
    """
    pixels = image_size[0] * image_size[1]
    
    # Base time for image processing
    base_time = max(2, pixels / 1000000)  # 2 seconds minimum
    
    # Add time for AI generation
    if has_ai_generation:
        base_time += 30  # Add 30 seconds for AI generation
    
    return int(base_time)

class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step: int = None, description: str = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if description:
            self.description = description
        
        progress = self.current_step / self.total_steps
        elapsed = time.time() - self.start_time
        
        if progress > 0:
            eta = (elapsed / progress) - elapsed
            logger.info(f"{self.description}: {progress:.1%} complete, ETA: {eta:.1f}s")
    
    def finish(self):
        """Mark as finished."""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed in {elapsed:.1f}s")

def log_system_info():
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {psutil.cpu_count()} cores")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        logger.warning("PyTorch not installed")
    
    logger.info("==========================")
