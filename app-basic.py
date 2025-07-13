"""
Basic Headshot Generator - Streamlit Cloud Compatible Version
This version works without face detection for maximum compatibility
"""

import streamlit as st
import os
from io import BytesIO
import logging
from PIL import Image
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="AI Headshot Generator (Basic)",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_image(image_file):
    """Load and validate uploaded image"""
    try:
        img = Image.open(image_file)
        return img
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def resize_image(image, max_size=1024):
    """Resize image while maintaining aspect ratio"""
    try:
        img = image.copy()
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image

def basic_enhancement(image):
    """Basic image enhancement without AI"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Basic enhancements
        # Brightness and contrast adjustment
        enhanced = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)
        
        # Slight blur for smoothing
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(enhanced)
    except Exception as e:
        logger.error(f"Error in basic enhancement: {str(e)}")
        return image

def main():
    st.title("🤖 AI Headshot Generator (Basic)")
    st.markdown("### Professional headshots made simple")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        enhancement_level = st.slider(
            "Enhancement Level",
            min_value=0,
            max_value=100,
            value=50,
            help="Adjust the level of image enhancement"
        )
        
        output_quality = st.selectbox(
            "Output Quality",
            ["High", "Medium", "Low"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a clear photo
        2. Adjust enhancement settings
        3. Download your professional headshot
        """)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Photo")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear photo for best results"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            original_image = load_image(uploaded_file)
            if original_image:
                st.image(original_image, caption="Original Photo", use_column_width=True)
                
                # Image info
                st.info(f"📊 Image: {original_image.size[0]}x{original_image.size[1]} pixels")
    
    with col2:
        st.header("✨ Enhanced Headshot")
        
        if uploaded_file is not None and 'original_image' in locals():
            if st.button("🎨 Generate Headshot", type="primary"):
                with st.spinner("Enhancing your photo..."):
                    try:
                        # Resize image
                        resized_image = resize_image(original_image)
                        
                        # Apply basic enhancement
                        enhanced_image = basic_enhancement(resized_image)
                        
                        # Display result
                        st.image(enhanced_image, caption="Professional Headshot", use_column_width=True)
                        
                        # Download button
                        img_buffer = BytesIO()
                        quality_map = {"High": 95, "Medium": 85, "Low": 75}
                        enhanced_image.save(
                            img_buffer, 
                            format="JPEG", 
                            quality=quality_map[output_quality]
                        )
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Headshot",
                            data=img_buffer.getvalue(),
                            file_name="professional_headshot.jpg",
                            mime="image/jpeg"
                        )
                        
                        st.success("✅ Headshot generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating headshot: {str(e)}")
                        logger.error(f"Generation error: {str(e)}")
        else:
            st.info("👆 Upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Basic AI Headshot Generator | Streamlit Cloud Compatible</p>
        <p>For advanced features with face detection, use the full version locally</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
