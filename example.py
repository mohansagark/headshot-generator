#!/usr/bin/env python3
"""
Example usage of the Professional Headshot Generator.
This script demonstrates the basic functionality without requiring heavy AI models.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_image(output_path: str = "sample_photo.jpg"):
    """Create a sample image for testing if no photo is available."""
    # Create a sample image with a simple face-like shape
    width, height = 512, 512
    image = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple face
    # Head (circle)
    draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], fill='peachpuff', outline='black', width=2)
    
    # Eyes
    eye_y = height//2 - 50
    draw.ellipse([width//2 - 80, eye_y, width//2 - 40, eye_y + 40], fill='white', outline='black')
    draw.ellipse([width//2 + 40, eye_y, width//2 + 80, eye_y + 40], fill='white', outline='black')
    draw.ellipse([width//2 - 70, eye_y + 10, width//2 - 50, eye_y + 30], fill='black')
    draw.ellipse([width//2 + 50, eye_y + 10, width//2 + 70, eye_y + 30], fill='black')
    
    # Nose
    nose_points = [(width//2, height//2), (width//2 - 10, height//2 + 30), (width//2 + 10, height//2 + 30)]
    draw.polygon(nose_points, fill='peachpuff', outline='black')
    
    # Mouth
    draw.arc([width//2 - 40, height//2 + 20, width//2 + 40, height//2 + 60], 0, 180, fill='black', width=3)
    
    image.save(output_path)
    logger.info(f"Sample image created: {output_path}")
    return output_path

def basic_demo():
    """Run a basic demo without heavy AI models."""
    print("🎭 Professional Headshot Generator - Basic Demo")
    print("=" * 50)
    
    # Check if we have an input image
    sample_image = None
    common_names = ['photo.jpg', 'photo.png', 'selfie.jpg', 'selfie.png', 'image.jpg', 'image.png']
    
    for name in common_names:
        if os.path.exists(name):
            sample_image = name
            break
    
    if not sample_image:
        print("📸 No input image found. Creating a sample image...")
        sample_image = create_sample_image()
    else:
        print(f"📸 Using existing image: {sample_image}")
    
    try:
        # Basic image processing without AI models
        from PIL import Image, ImageEnhance, ImageFilter
        
        print("🔄 Processing image...")
        
        # Load and process the image
        with Image.open(sample_image) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            print(f"   Original size: {img.size}")
            
            # Resize if too large
            max_size = 512
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"   Resized to: {img.size}")
            
            # Apply basic enhancements
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            print("   ✅ Applied brightness, contrast, and sharpness enhancements")
            
            # Create a simple professional background
            bg_img = Image.new('RGB', img.size, color=(240, 240, 240))
            
            # Composite the enhanced image on the background
            if img.mode == 'RGBA':
                bg_img.paste(img, (0, 0), img)
            else:
                bg_img = img
            
            # Save the result
            output_path = "demo_output.jpg"
            bg_img.save(output_path, quality=95)
            
            print(f"✨ Enhanced image saved as: {output_path}")
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return
    
    print("\n🎉 Basic demo completed!")
    print("📝 To use the full AI features:")
    print("   1. Install all dependencies: pip install -r requirements.txt")
    print("   2. Run the web interface: streamlit run app.py")
    print("   3. Or use the CLI: python cli.py process your_photo.jpg")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_basic = ['PIL', 'numpy']
    optional_ai = ['torch', 'diffusers', 'transformers', 'cv2', 'mediapipe', 'rembg']
    
    missing_basic = []
    missing_ai = []
    
    for module in required_basic:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_basic.append(module)
            print(f"   ❌ {module}")
    
    for module in optional_ai:
        try:
            __import__(module)
            print(f"   ✅ {module} (AI features available)")
        except ImportError:
            missing_ai.append(module)
            print(f"   ⚠️  {module} (install for AI features)")
    
    if missing_basic:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_basic)}")
        print("   Run: pip install Pillow numpy")
        return False
    
    if missing_ai:
        print(f"\n⚠️  AI features not available. Missing: {', '.join(missing_ai)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("\n🎉 All dependencies installed! Full AI features available.")
    
    return True

if __name__ == "__main__":
    print("🚀 Professional Headshot Generator - Example Demo")
    print("This is a lightweight demo to test basic functionality.\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print()
    basic_demo()
    
    print("\n📚 Next Steps:")
    print("   • Try the web interface: streamlit run app.py")
    print("   • Use CLI for batch processing: python cli.py batch input/ output/")
    print("   • Generate AI headshots: python cli.py generate 'professional headshot'")
    print("   • Read the README.md for complete documentation")
