import streamlit as st
import os
from PIL import Image
import io
import base64
from headshot_generator import HeadshotGenerator
import gc

# Check deployment environment and show compatibility info
def check_deployment_compatibility():
    """Check if running in cloud deployment and show compatibility warnings"""
    cloud_platform = None
    if "STREAMLIT_CLOUD" in os.environ:
        cloud_platform = "Streamlit Cloud"
    elif "DYNO" in os.environ or "HEROKU_SLUG_COMMIT" in os.environ:
        cloud_platform = "Heroku"
    elif "RAILWAY_ENVIRONMENT" in os.environ:
        cloud_platform = "Railway"
    
    if cloud_platform:
        st.info(f"🌐 Running on {cloud_platform} - Some advanced face detection features may be limited for compatibility")
        return True
    return False

# Deployment optimizations
if "DYNO" in os.environ or "HEROKU_SLUG_COMMIT" in os.environ:
    # Running on Heroku
    os.environ["TORCH_HOME"] = "/tmp/.torch"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.transformers"
    os.environ["DIFFUSERS_CACHE"] = "/tmp/.diffusers"
elif "STREAMLIT_CLOUD" in os.environ:
    # Running on Streamlit Cloud
    os.environ["TORCH_HOME"] = "/tmp/.torch"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.transformers"
    os.environ["DIFFUSERS_CACHE"] = "/tmp/.diffusers"

# Configure page
st.set_page_config(
    page_title="Professional Headshot Generator",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_generator():
    """Load the headshot generator with caching for better performance."""
    try:
        generator = HeadshotGenerator()
        # Test basic functionality
        if hasattr(generator, 'pipeline') and generator.pipeline is None:
            st.warning("⚠️ AI generation features not available in this deployment. Basic image processing will work.")
        return generator
    except Exception as e:
        st.error(f"Failed to initialize generator: {str(e)}")
        # Return a minimal generator that can handle basic operations
        return None

def download_image(image, filename):
    """Create a download button for the processed image."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    st.download_button(
        label="📥 Download Professional Headshot",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        key=f"download_{filename}"
    )

def main():
    # Check deployment compatibility
    check_deployment_compatibility()
    
    # Header
    st.markdown('<h1 class="main-header">📸 Professional Headshot Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your photos into professional headshots using AI</p>', unsafe_allow_html=True)
    
    # Initialize the generator
    generator = load_generator()
    
    if generator is None:
        st.error("❌ Could not initialize the headshot generator")
        st.info("💡 This may be due to missing dependencies or memory limitations in the deployment environment.")
        st.info("🔄 Try refreshing the page or contact support if the issue persists.")
        return
    
    st.success("✅ Headshot generator initialized successfully!")
    
    # Sidebar for options
    st.sidebar.title("🎛️ Settings")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["📷 Process Existing Photo", "🎨 Generate from Text", "📁 Batch Processing"]
    )
    
    if mode == "📷 Process Existing Photo":
        st.header("📷 Upload Your Photo")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear photo of yourself for processing"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Photo")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)
            
            # Processing options
            st.sidebar.subheader("🎨 Processing Options")
            
            remove_bg = st.sidebar.checkbox("Remove Background", value=True)
            enhance_image = st.sidebar.checkbox("Enhance Image Quality", value=True)
            add_bg = st.sidebar.checkbox("Add Professional Background", value=True)
            
            if add_bg:
                bg_type = st.sidebar.selectbox(
                    "Background Type",
                    ["gradient", "solid", "blur"]
                )
                
                if bg_type in ["gradient", "solid"]:
                    bg_color = st.sidebar.color_picker("Background Color", "#f0f0f0")
                    # Convert hex to RGB
                    bg_color_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
            
            # Enhancement settings
            if enhance_image:
                st.sidebar.subheader("✨ Enhancement Settings")
                brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.1, 0.1)
                contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.1, 0.1)
                sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.2, 0.1)
                color_sat = st.sidebar.slider("Color Saturation", 0.5, 2.0, 1.1, 0.1)
            
            # Process button
            if st.button("🚀 Generate Professional Headshot", type="primary"):
                with st.spinner("🔄 Processing your photo... This may take a few moments."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = "temp_upload.png"  # Use PNG to preserve transparency
                        original_image.save(temp_path)
                        
                        # Process the image
                        processed_image = generator.process_existing_photo(
                            temp_path,
                            remove_bg=remove_bg,
                            enhance=enhance_image,
                            add_background=add_bg,
                            background_type=bg_type if add_bg else "gradient"
                        )
                        
                        # Apply custom enhancements if enabled
                        if enhance_image:
                            processed_image = generator.enhance_image(
                                processed_image,
                                brightness=brightness,
                                contrast=contrast,
                                sharpness=sharpness,
                                color=color_sat
                            )
                        
                        # Add custom background if needed
                        if add_bg and bg_type in ["gradient", "solid"]:
                            processed_image = generator.add_professional_background(
                                processed_image,
                                background_type=bg_type,
                                color1=bg_color_rgb,
                                color2=bg_color_rgb
                            )
                        
                        # Display result
                        with col2:
                            st.subheader("✨ Professional Headshot")
                            st.image(processed_image, use_container_width=True)
                            
                            # Download button
                            download_image(processed_image, "professional_headshot.png")
                        
                        # Clean up
                        os.remove(temp_path)
                        
                        st.success("🎉 Your professional headshot is ready!")
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")
    
    elif mode == "🎨 Generate from Text":
        st.header("🎨 Generate Headshot from Description")
        
        # Check if Stable Diffusion is available
        if not hasattr(generator, 'sd_pipeline') or generator.sd_pipeline is None:
            st.info("🔄 Loading Stable Diffusion model... This may take a few minutes on first run.")
            
            if st.button("🚀 Load AI Model"):
                with st.spinner("Loading Stable Diffusion model..."):
                    try:
                        generator.load_stable_diffusion()
                        st.success("✅ AI model loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {e}")
                        return
        else:
            # Preset professional prompts
            st.subheader("🎯 Quick Presets")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👔 Executive Style", use_container_width=True):
                    st.session_state.prompt = "professional executive corporate headshot, middle-aged business person wearing dark navy blue tailored suit, crisp white dress shirt, silk tie, confident professional expression, direct eye contact, subtle smile, soft studio lighting with key light and fill light, blurred modern glass office background with warm bokeh lights, shallow depth of field, shot with 85mm lens"
                    st.rerun()
            
            with col2:
                if st.button("💼 Business Casual", use_container_width=True):
                    st.session_state.prompt = "approachable professional business headshot, person wearing charcoal gray blazer, light blue dress shirt, no tie, open collar, friendly confident smile, warm eye contact, natural window lighting, soft shadows, blurred contemporary office workspace background, professional but relaxed atmosphere"
                    st.rerun()
            
            with col3:
                if st.button("🏢 Corporate Formal", use_container_width=True):
                    st.session_state.prompt = "formal corporate executive headshot, person wearing premium black business suit, pristine white dress shirt, conservative dark tie, serious professional demeanor, authoritative presence, perfect studio lighting setup, elegant blurred mahogany office interior background with soft ambient lighting, executive portrait photography style"
                    st.rerun()
            
            # Text input for prompt
            prompt = st.text_area(
                "Describe your desired headshot:",
                value=st.session_state.get("prompt", "professional corporate headshot of a business person wearing a dark navy suit, crisp white dress shirt, professional tie, confident expression, soft professional lighting, blurred modern office background with bokeh effect, shallow depth of field, high quality portrait photography, 8k resolution"),
                height=120,
                help="Describe the type of headshot you want. Be specific about clothing, lighting, and style."
            )
            
            negative_prompt = st.text_area(
                "What to avoid (negative prompt):",
                value="low quality, blurry face, distorted features, deformed hands, extra limbs, bad anatomy, casual clothing, t-shirt, hoodie, messy background, cluttered office, harsh shadows, overexposed, underexposed, amateur photography, cartoon, anime, painting",
                height=100
            )
            
            # Generation settings
            st.sidebar.subheader("🎛️ Generation Settings")
            width = st.sidebar.slider("Width", 256, 1024, 768, 64)
            height = st.sidebar.slider("Height", 256, 1024, 768, 64)
            steps = st.sidebar.slider("Quality (steps)", 30, 150, 80, 10)
            guidance = st.sidebar.slider("Prompt Adherence", 3.0, 15.0, 9.0, 0.5)
            
            # Advanced settings
            st.sidebar.subheader("🔧 Advanced Settings")
            use_seed = st.sidebar.checkbox("Use Custom Seed (for reproducible results)")
            seed = None
            if use_seed:
                seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
            
            if st.button("🎨 Generate Headshot", type="primary"):
                with st.spinner("🔄 Generating your headshot... This may take a few minutes."):
                    try:
                        generated_image = generator.generate_headshot_from_prompt(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            seed=seed
                        )
                        
                        # Display result
                        st.subheader("✨ Generated Headshot")
                        st.image(generated_image, use_container_width=True)
                        
                        # Download button
                        download_image(generated_image, "generated_headshot.png")
                        
                        st.success("🎉 Your AI-generated headshot is ready!")
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
    
    elif mode == "📁 Batch Processing":
        st.header("📁 Batch Process Multiple Photos")
        
        st.info("💡 Upload multiple photos to process them all at once with the same settings.")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple photos for batch processing"
        )
        
        if uploaded_files:
            st.write(f"📊 Selected {len(uploaded_files)} files for processing")
            
            # Processing options
            st.sidebar.subheader("🎨 Batch Processing Options")
            remove_bg = st.sidebar.checkbox("Remove Background", value=True, key="batch_remove_bg")
            enhance_image = st.sidebar.checkbox("Enhance Image Quality", value=True, key="batch_enhance")
            add_bg = st.sidebar.checkbox("Add Professional Background", value=True, key="batch_add_bg")
            
            if st.button("🚀 Process All Photos", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.spinner(f"🔄 Processing {uploaded_file.name}..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = f"temp_batch_{i}.png"  # Use PNG to preserve transparency
                            image = Image.open(uploaded_file)
                            image.save(temp_path)
                            
                            # Process the image
                            processed_image = generator.process_existing_photo(
                                temp_path,
                                remove_bg=remove_bg,
                                enhance=enhance_image,
                                add_background=add_bg
                            )
                            
                            results.append((uploaded_file.name, processed_image))
                            
                            # Clean up
                            os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                if results:
                    st.success(f"🎉 Successfully processed {len(results)} photos!")
                    
                    # Create columns for display
                    cols = st.columns(min(3, len(results)))
                    
                    for i, (filename, processed_image) in enumerate(results):
                        with cols[i % len(cols)]:
                            st.subheader(f"✨ {filename}")
                            st.image(processed_image, use_container_width=True)
                            download_image(processed_image, f"professional_{filename}")
    
    # Footer with tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📸 Photo Quality**
        - Use high-resolution images
        - Ensure good lighting
        - Face should be clearly visible
        - Avoid heavy shadows
        """)
    
    with col2:
        st.markdown("""
        **🎨 Processing Tips**
        - Remove background for cleaner results
        - Enhance image quality for professional look
        - Choose appropriate background style
        - Adjust enhancement settings as needed
        """)
    
    with col3:
        st.markdown("""
        **🚀 AI Generation**
        - Be specific in your descriptions
        - Mention professional attire
        - Include lighting preferences
        - Use negative prompts to avoid issues
        """)

if __name__ == "__main__":
    # Check deployment compatibility
    check_deployment_compatibility()
    
    main()

    # Memory management
    gc.collect()
