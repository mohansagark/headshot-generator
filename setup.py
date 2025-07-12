#!/usr/bin/env python3
"""
Setup script for the Professional Headshot Generator.
This script helps users set up the environment and verify installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available memory: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM available. AI features may be slow.")
        elif memory_gb >= 8:
            print("✅ Sufficient memory for optimal performance")
        else:
            print("✅ Adequate memory for basic AI features")
            
    except ImportError:
        print("⚠️  Could not check memory (psutil not installed)")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA GPU available: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            print("🚀 Apple Silicon GPU (MPS) available")
        else:
            print("💻 Using CPU (GPU acceleration not available)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")
    
    return True

def setup_virtual_environment():
    """Set up Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment
    python_cmd = sys.executable
    if not run_command(f'"{python_cmd}" -m venv venv', "Creating virtual environment"):
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies."""
    
    # Determine pip command
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install basic dependencies first
    basic_deps = [
        "torch", "torchvision", "numpy", "Pillow", "opencv-python",
        "streamlit", "fastapi", "uvicorn", "python-dotenv", "tqdm"
    ]
    
    for dep in basic_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    # Install AI dependencies
    ai_deps = [
        "diffusers", "transformers", "accelerate", "mediapipe",
        "rembg", "gradio", "scikit-image"
    ]
    
    print("\n🧠 Installing AI dependencies (this may take a while)...")
    for dep in ai_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, some AI features may not work")
    
    # Try to install xformers for better performance
    print("\n🚀 Installing performance optimizations...")
    run_command(f"{pip_cmd} install xformers", "Installing xformers (optional)")
    
    return True

def verify_installation():
    """Verify that the installation was successful."""
    print("\n🔍 Verifying installation...")
    
    # Determine python command
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Test basic imports
    test_script = '''
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"PyTorch error: {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV error: {e}")

try:
    import streamlit
    print(f"Streamlit: {streamlit.__version__}")
except ImportError as e:
    print(f"Streamlit error: {e}")

try:
    from diffusers import StableDiffusionPipeline
    print("Diffusers: Available")
except ImportError as e:
    print(f"Diffusers error: {e}")

try:
    import mediapipe
    print("MediaPipe: Available")
except ImportError as e:
    print(f"MediaPipe error: {e}")

try:
    import rembg
    print("rembg: Available")
except ImportError as e:
    print(f"rembg error: {e}")

print("\\n✅ Verification complete!")
'''
    
    return run_command(f'"{python_cmd}" -c "{test_script}"', "Running verification tests")

def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            content = env_example.read_text()
            env_file.write_text(content)
            print("✅ Created .env file from .env.example")
            return True
        except Exception as e:
            print(f"⚠️  Could not create .env file: {e}")
    
    return True

def main():
    """Main setup function."""
    print("🎭 Professional Headshot Generator - Setup Script")
    print("=" * 60)
    print("This script will help you set up the headshot generator environment.\n")
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please address the issues above.")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print("\n❌ Failed to set up virtual environment.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Some dependencies failed to install. The app may have limited functionality.")
    
    # Create .env file
    create_sample_env()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with some issues. Please check the errors above.")
    
    print("\n📚 Next Steps:")
    print("1. 🧪 Test the installation:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\python example.py")
    else:
        print("   venv/bin/python example.py")
    
    print("\n2. 🌐 Start the web interface:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\python -m streamlit run app.py")
    else:
        print("   venv/bin/python -m streamlit run app.py")
    
    print("\n3. 💻 Use the command line:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\python cli.py process your_photo.jpg")
    else:
        print("   venv/bin/python cli.py process your_photo.jpg")
    
    print("\n4. 📖 Read the documentation:")
    print("   Check README.md for detailed usage instructions")
    
    print("\n✨ Happy headshot generating!")

if __name__ == "__main__":
    main()
