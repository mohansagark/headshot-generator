<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Professional Headshot Generator - Copilot Instructions

## Project Overview

This is a Python-based AI application for generating professional headshots using generative AI models and advanced image processing techniques.

## Key Technologies

- **AI Models**: Stable Diffusion, PyTorch, Transformers, Diffusers
- **Image Processing**: OpenCV, PIL/Pillow, MediaPipe, scikit-image
- **Face Detection**: MediaPipe, dlib, face-recognition, InsightFace
- **Background Removal**: rembg, backgroundremover
- **Web Interface**: Streamlit, Gradio, FastAPI
- **CLI**: argparse, pathlib

## Code Style Guidelines

- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Include comprehensive docstrings for all classes and methods
- Add logging for important operations and error handling
- Use meaningful variable and function names
- Handle exceptions gracefully with user-friendly error messages

## Architecture Patterns

- Use dependency injection for AI model loading
- Implement caching for expensive operations (model loading, image processing)
- Separate concerns: core logic in `headshot_generator.py`, UI in `app.py`, CLI in `cli.py`
- Use factory patterns for creating different types of backgrounds and enhancements

## AI Model Integration

- Always check device availability (CUDA, MPS, CPU) before loading models
- Implement memory-efficient loading with proper cleanup
- Use context managers for model inference to manage GPU memory
- Support multiple model backends and fallback options

## Image Processing Best Practices

- Preserve image quality throughout the processing pipeline
- Support multiple image formats (PNG, JPG, JPEG, BMP, TIFF)
- Implement batch processing for efficiency
- Use proper color space conversions
- Handle edge cases (no face detected, corrupted images, etc.)

## Error Handling

- Provide clear, actionable error messages to users
- Log detailed error information for debugging
- Implement graceful degradation when AI models fail to load
- Validate input parameters and file formats

## Performance Considerations

- Use lazy loading for AI models
- Implement progress indicators for long-running operations
- Optimize image processing pipelines
- Cache processed results when appropriate
- Use multiprocessing for batch operations

## Security Guidelines

- Validate all user inputs, especially file uploads
- Sanitize file paths and names
- Limit file sizes and types for uploads
- Don't store sensitive user data permanently

## Testing Recommendations

- Test with various image sizes and formats
- Verify face detection accuracy across different demographics
- Test background removal with complex scenes
- Validate AI model outputs for quality and appropriateness
- Test CLI commands with edge cases

## Documentation Standards

- Keep README.md updated with current features and setup instructions
- Document all configuration options and parameters
- Provide clear usage examples for both CLI and web interface
- Include troubleshooting guide for common issues

## Dependencies Management

- Pin major versions in requirements.txt for stability
- Group dependencies by functionality (AI, image processing, web, etc.)
- Test with latest versions before updating
- Document system requirements and compatibility
