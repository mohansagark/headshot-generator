# 📸 Professional Headshot Generator

An AI-powered application that transforms your photos into professional headshots using advanced machine learning models and image processing techniques.

## ✨ Features

### 🎯 Core Capabilities

- **AI-Powered Enhancement**: Uses state-of-the-art machine learning models for professional-quality results
- **Background Removal**: Automatically removes and replaces backgrounds with professional alternatives
- **Face Detection**: Advanced face detection and positioning for optimal headshot composition
- **Image Enhancement**: Automatic brightness, contrast, sharpness, and color correction
- **Multiple Interfaces**: Web UI, command-line interface, and API support

### 🎨 Processing Options

- **Background Types**: Gradient, solid color, or blurred backgrounds
- **Enhancement Controls**: Adjustable brightness, contrast, sharpness, and saturation
- **Batch Processing**: Process multiple photos simultaneously
- **Custom Styling**: Professional lighting and color adjustments
- **High Quality Output**: Support for high-resolution images

### 🚀 AI Generation

- **Text-to-Image**: Generate headshots from text descriptions using Stable Diffusion
- **Professional Prompts**: Pre-configured prompts for business and professional use
- **Customizable Parameters**: Control image quality, style, and composition
- **Multiple Models**: Support for different AI models and styles

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for AI generation)
- GPU with CUDA support (optional, for faster processing)

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/headshot-generator.git
   cd headshot-generator
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment (optional)**
   ```bash
   cp .env.example .env
   # Edit .env file with your preferences
   ```

## 🚀 Usage

### Web Interface (Recommended)

Start the Streamlit web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and enjoy the intuitive interface for:

- Uploading and processing photos
- Generating headshots from text descriptions
- Batch processing multiple images
- Real-time parameter adjustment

### Command Line Interface

Process a single photo:

```bash
python cli.py process photo.jpg --output professional_headshot.jpg
```

Generate from text prompt:

```bash
python cli.py generate "professional headshot, business attire, studio lighting" --output generated.jpg
```

Batch process a directory:

```bash
python cli.py batch input_photos/ output_photos/
```

### Advanced CLI Examples

Custom background and enhancements:

```bash
python cli.py process photo.jpg \
  --background-type gradient \
  --bg-color "240,240,240" \
  --brightness 1.2 \
  --contrast 1.1 \
  --sharpness 1.3
```

High-quality AI generation:

```bash
python cli.py generate \
  "professional corporate headshot, confident expression, navy blue suit" \
  --width 768 \
  --height 768 \
  --steps 100 \
  --guidance 8.0
```

## 🎛️ Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`) to customize:

```env
# AI Model Settings
DEVICE=auto  # cuda, mps, cpu, or auto
STABLE_DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5

# Image Processing
DEFAULT_OUTPUT_SIZE=512
MAX_IMAGE_SIZE=2048
JPEG_QUALITY=95

# Performance
USE_HALF_PRECISION=true
ENABLE_XFORMERS=true
LOW_MEMORY_MODE=false
```

### Supported Devices

- **CUDA**: NVIDIA GPUs (fastest)
- **MPS**: Apple Silicon Macs (M1/M2)
- **CPU**: Universal fallback (slower)

## 📋 API Reference

### HeadshotGenerator Class

```python
from headshot_generator import HeadshotGenerator

# Initialize
generator = HeadshotGenerator(device='auto')

# Process existing photo
result = generator.process_existing_photo(
    'photo.jpg',
    remove_bg=True,
    enhance=True,
    add_background=True
)

# Generate from prompt
generator.load_stable_diffusion()
generated = generator.generate_headshot_from_prompt(
    "professional headshot, business attire"
)
```

### Key Methods

- `process_existing_photo()`: Process uploaded photos
- `generate_headshot_from_prompt()`: AI generation from text
- `remove_background()`: Background removal
- `enhance_image()`: Quality enhancement
- `detect_face()`: Face detection
- `batch_process()`: Multiple image processing

## 🎨 Tips for Best Results

### 📸 Photo Quality

- Use high-resolution images (1024px+ recommended)
- Ensure good lighting with minimal shadows
- Face should be clearly visible and centered
- Avoid heavily compressed or blurry images

### 🎭 AI Generation

- Be specific in descriptions: "professional headshot, business attire, studio lighting"
- Include style details: "confident expression, neutral background"
- Use negative prompts to avoid unwanted elements
- Experiment with different guidance scales (7.5-15.0)

### 🖼️ Background Selection

- **Gradient**: Professional and versatile
- **Solid**: Clean and corporate
- **Blur**: Maintains original context while reducing distractions

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Errors**

```bash
# Enable low memory mode
export LOW_MEMORY_MODE=true
# Or use CPU
export DEVICE=cpu
```

**2. Model Loading Errors**

```bash
# Clear cache and retry
rm -rf ./cache
python -c "import torch; torch.hub.clear_cache()"
```

**3. Face Detection Issues**

- Ensure face is clearly visible
- Check image orientation
- Try adjusting `MIN_DETECTION_CONFIDENCE` in config

**4. Slow Processing**

- Use GPU if available (`DEVICE=cuda` or `DEVICE=mps`)
- Enable half precision (`USE_HALF_PRECISION=true`)
- Reduce image size for testing

### Performance Optimization

For faster processing:

1. Use GPU acceleration (CUDA/MPS)
2. Enable xformers memory efficient attention
3. Use half precision (FP16)
4. Process images in batches
5. Optimize image sizes

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/headshot-generator.git
cd headshot-generator

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 🚀 Cloud Deployment

### Streamlit Cloud Deployment

The application is optimized for deployment on Streamlit Cloud with a minimal dependency set:

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Connect your GitHub account** and select this repository

4. **Set the main file path** to `app.py`

5. **Advanced settings**:
   - Python version: `3.11`
   - Requirements file: `requirements-deploy.txt`

### Deployment Features

- **Automatic dependency optimization**: Uses `requirements-deploy.txt` for cloud compatibility
- **Memory management**: Optimized for cloud resource limits
- **Background processing**: Efficient model loading and caching
- **Error handling**: Graceful fallbacks when advanced features aren't available

## 🐛 Troubleshooting

### Common Deployment Issues

#### "CMake not found" or "dlib installation failed"

**Problem**: Face detection libraries (dlib, mediapipe) require system dependencies not available in cloud environments.

**Solution**: The app automatically falls back to basic processing without face detection. This is expected behavior on cloud platforms.

#### "Out of memory" errors

**Solutions**:

- Use the deployment-optimized requirements: `requirements-deploy.txt`
- Reduce image sizes before processing
- Process images one at a time instead of batch processing

#### "Model loading timeout"

**Solutions**:

- Use smaller AI models (configured automatically in deployment)
- Increase memory allocation if using a paid cloud service
- Be patient during first load - models are cached after initial download

### Local Development Issues

#### GPU/CUDA Issues

```bash
# For CUDA support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Face Detection Issues

```bash
# If mediapipe installation fails
pip install mediapipe --no-deps
pip install opencv-python

# Alternative: use CPU-only version
pip install mediapipe-cpu
```

#### Memory Issues

```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://stability.ai/) for text-to-image generation
- [MediaPipe](https://mediapipe.dev/) for face detection
- [rembg](https://github.com/danielgatis/rembg) for background removal
- [Streamlit](https://streamlit.io/) for the web interface
- The open-source community for various tools and libraries

## 📞 Support

- 📚 [Documentation](https://github.com/yourusername/headshot-generator/wiki)
- 🐛 [Bug Reports](https://github.com/yourusername/headshot-generator/issues)
- 💬 [Discussions](https://github.com/yourusername/headshot-generator/discussions)
- 📧 Email: support@yourproject.com

---

**Made with ❤️ for creating professional headshots using AI**
