import os
import logging
from typing import Optional, Tuple, List
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. AI generation features will be disabled.")
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available. Some image processing features may be limited.")
    CV2_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    logger.warning("Diffusers not available. AI generation features will be disabled.")
    DIFFUSERS_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    logger.warning("rembg not available. Background removal features will be limited.")
    REMBG_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not available. Face detection features will be limited.")
    MEDIAPIPE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeadshotGenerator:
    """
    Professional headshot generator using AI models and image processing techniques.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the headshot generator with AI models and processors.
        
        Args:
            device: Device to run models on ('cuda', 'mps', or 'cpu')
        """
        self.device = device or self._get_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize face detection if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        else:
            self.face_detection = None
            logger.warning("Face detection not available - MediaPipe not installed")
        
        # Initialize Stable Diffusion pipeline
        self.sd_pipeline = None
        
    def _get_device(self) -> str:
        """Automatically detect the best available device."""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_stable_diffusion(self, model_id: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE"):
        """
        Load Stable Diffusion model for image generation.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion (default: Realistic Vision for better portraits)
        """
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch not available. Cannot load Stable Diffusion model.")
            
        if not DIFFUSERS_AVAILABLE:
            raise ValueError("Diffusers not available. Cannot load Stable Diffusion model.")
            
        try:
            logger.info(f"Loading Stable Diffusion model: {model_id}")
            
            # Try to load the realistic vision model first, fallback to standard if needed
            try:
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            except Exception as e:
                logger.warning(f"Failed to load {model_id}, falling back to standard model: {e}")
                # Fallback to standard model
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Use DDIM scheduler for better quality and consistency
            self.sd_pipeline.scheduler = DDIMScheduler.from_config(
                self.sd_pipeline.scheduler.config
            )
            self.sd_pipeline = self.sd_pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.device != "cpu":
                try:
                    self.sd_pipeline.enable_model_cpu_offload()
                except Exception:
                    pass
                    
            if hasattr(self.sd_pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self.sd_pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
                    
            # Enable VAE slicing for memory efficiency
            if hasattr(self.sd_pipeline, "enable_vae_slicing"):
                self.sd_pipeline.enable_vae_slicing()
                    
            logger.info("Stable Diffusion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            raise
    
    def detect_face(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in the image and return bounding box.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (x, y, width, height) or None if no face detected
        """
        if not MEDIAPIPE_AVAILABLE or not self.face_detection:
            logger.warning("Face detection not available - using center crop fallback")
            # Return a center crop as fallback
            w, h = image.size
            size = min(w, h)
            x = (w - size) // 2
            y = (h - size) // 2
            return (x, y, size, size)
        
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for face detection")
            return None
            
        # Convert PIL to cv2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            detection = results.detections[0]  # Use first detection
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = rgb_image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            return (x, y, width, height)
        
        return None
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background from the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Image with transparent background
        """
        if not REMBG_AVAILABLE:
            logger.warning("Background removal not available - rembg not installed")
            # Return original image with alpha channel as fallback
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            return image
            
        try:
            logger.info("Removing background...")
            # Convert to bytes for rembg
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Remove background
            result = remove(img_byte_arr)
            
            # Convert back to PIL
            return Image.open(io.BytesIO(result))
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return image
    
    def enhance_image(self, image: Image.Image, 
                     brightness: float = 1.1,
                     contrast: float = 1.1,
                     sharpness: float = 1.2,
                     color: float = 1.1) -> Image.Image:
        """
        Enhance image quality with adjustable parameters.
        
        Args:
            image: PIL Image
            brightness: Brightness factor (1.0 = original)
            contrast: Contrast factor (1.0 = original)
            sharpness: Sharpness factor (1.0 = original)
            color: Color saturation factor (1.0 = original)
            
        Returns:
            Enhanced PIL Image
        """
        # Apply enhancements
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color)
        
        return image
    
    def add_professional_background(self, image: Image.Image, 
                                  background_type: str = "gradient",
                                  color1: Tuple[int, int, int] = (240, 240, 240),
                                  color2: Tuple[int, int, int] = (200, 200, 200)) -> Image.Image:
        """
        Add a professional background to the image.
        
        Args:
            image: PIL Image with transparent background
            background_type: Type of background ('gradient', 'solid', 'blur')
            color1: Primary background color
            color2: Secondary background color (for gradients)
            
        Returns:
            Image with professional background
        """
        width, height = image.size
        
        if background_type == "gradient":
            # Create gradient background
            background = Image.new('RGB', (width, height))
            for y in range(height):
                r = int(color1[0] + (color2[0] - color1[0]) * y / height)
                g = int(color1[1] + (color2[1] - color1[1]) * y / height)
                b = int(color1[2] + (color2[2] - color1[2]) * y / height)
                for x in range(width):
                    background.putpixel((x, y), (r, g, b))
        elif background_type == "solid":
            background = Image.new('RGB', (width, height), color1)
        else:  # blur
            # Create a blurred version of the original as background
            background = image.convert('RGB').filter(ImageFilter.GaussianBlur(radius=20))
        
        # Composite the image onto the background
        if image.mode == 'RGBA':
            background.paste(image, (0, 0), image)
        else:
            background.paste(image, (0, 0))
        
        return background
    
    def generate_headshot_from_prompt(self, prompt: str, 
                                    negative_prompt: str = "low quality, blurry, distorted, deformed",
                                    width: int = 768, height: int = 768,
                                    num_inference_steps: int = 80,
                                    guidance_scale: float = 9.0,
                                    seed: Optional[int] = None) -> Image.Image:
        """
        Generate a headshot from a text prompt using Stable Diffusion.
        
        Args:
            prompt: Text description of the desired headshot
            negative_prompt: What to avoid in the generation
            width: Output width
            height: Output height
            num_inference_steps: Number of denoising steps (higher = better quality)
            guidance_scale: How closely to follow the prompt (7-12 for realistic images)
            seed: Random seed for reproducible results
            
        Returns:
            Generated PIL Image
        """
        if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
            raise ValueError("AI generation not available. PyTorch and Diffusers are required.")
            
        if self.sd_pipeline is None:
            raise ValueError("Stable Diffusion pipeline not loaded. Call load_stable_diffusion() first.")
        
        # Enhanced prompt engineering for professional headshots
        enhanced_prompt = f"highly detailed professional corporate headshot portrait, {prompt}, sharp focus, professional photography, studio lighting, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, photorealistic"
        
        # Enhanced negative prompt for better quality
        enhanced_negative = f"{negative_prompt}, amateur, snapshot, phone camera, selfie, multiple people, hands visible, full body, cropped face, bad lighting, harsh shadows, overexposed, underexposed, noise, artifacts, jpeg compression, watermark, signature, text, logo, brand, cartoon, anime, 3d render, painting, drawing, sketch, illustration"
        
        logger.info(f"Generating professional headshot with enhanced prompt")
        
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.inference_mode():
            result = self.sd_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                eta=0.0,  # DDIM deterministic sampling
            )
        
        generated_image = result.images[0]
        
        # Post-process for better professional appearance
        generated_image = self.enhance_image(
            generated_image,
            brightness=1.05,
            contrast=1.1,
            sharpness=1.15,
            color=1.05
        )
        
        return generated_image
    
    def process_existing_photo(self, image_path: str,
                             remove_bg: bool = True,
                             enhance: bool = True,
                             add_background: bool = True,
                             background_type: str = "gradient") -> Image.Image:
        """
        Process an existing photo to create a professional headshot.
        
        Args:
            image_path: Path to the input image
            remove_bg: Whether to remove the background
            enhance: Whether to enhance the image quality
            add_background: Whether to add a professional background
            background_type: Type of background to add
            
        Returns:
            Processed PIL Image
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Processing image: {image_path}")
        
        # Detect face for validation
        face_bbox = self.detect_face(image)
        if face_bbox is None:
            logger.warning("No face detected in the image")
        else:
            logger.info("Face detected successfully")
        
        # Remove background if requested
        if remove_bg:
            image = self.remove_background(image)
        
        # Enhance image if requested
        if enhance:
            image = self.enhance_image(image)
        
        # Add professional background if requested
        if add_background and remove_bg:
            image = self.add_professional_background(image, background_type)
        
        # Ensure the final image is in a consistent format
        # If we didn't add a background and have transparency, keep RGBA
        # Otherwise, ensure RGB for compatibility
        if not (add_background and remove_bg) and image.mode == 'RGBA':
            # Keep transparency if no background was added
            pass
        else:
            # Convert to RGB for better compatibility
            image = self._ensure_rgb_for_jpeg(image)
        
        return image
    
    def batch_process(self, input_dir: str, output_dir: str, **kwargs) -> List[str]:
        """
        Process multiple images in batch.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            **kwargs: Additional arguments for process_existing_photo
            
        Returns:
            List of processed image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_files = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"headshot_{filename}")
                
                try:
                    processed_image = self.process_existing_photo(input_path, **kwargs)
                    processed_image.save(output_path)
                    processed_files.append(output_path)
                    logger.info(f"Processed: {filename}")
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
        
        return processed_files

    def _ensure_rgb_for_jpeg(self, image: Image.Image) -> Image.Image:
        """
        Convert RGBA images to RGB with white background for JPEG compatibility.
        
        Args:
            image: PIL Image that may have transparency
            
        Returns:
            RGB PIL Image suitable for JPEG saving
        """
        if image.mode in ('RGBA', 'LA'):
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(image, mask=image.split()[-1])  # Use transparency as mask
            return background
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = HeadshotGenerator()
    
    # Load Stable Diffusion model (optional, for text-to-image generation)
    # generator.load_stable_diffusion()
    
    # Example: Process an existing photo
    # result = generator.process_existing_photo("path/to/your/photo.jpg")
    # result.save("professional_headshot.jpg")
    
    # Example: Generate from prompt
    # generator.load_stable_diffusion()
    # prompt = "professional headshot of a person, business attire, studio lighting, high quality"
    # generated = generator.generate_headshot_from_prompt(prompt)
    # generated.save("generated_headshot.jpg")
    
    print("HeadshotGenerator initialized successfully!")
