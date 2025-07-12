#!/usr/bin/env python3
"""
Command-line interface for the Professional Headshot Generator.
"""

import argparse
import os
import sys
from pathlib import Path
from headshot_generator import HeadshotGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Professional Headshot Generator - Create professional headshots using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single photo
  python cli.py process photo.jpg --output headshot.jpg

  # Process with custom settings
  python cli.py process photo.jpg --no-remove-bg --enhance --brightness 1.2

  # Generate from text prompt
  python cli.py generate "professional headshot, business attire" --output generated.jpg

  # Batch process a directory
  python cli.py batch input_photos/ output_photos/

  # Use custom background
  python cli.py process photo.jpg --background-type solid --bg-color "240,240,240"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process an existing photo')
    process_parser.add_argument('input', help='Input image file path')
    process_parser.add_argument('-o', '--output', help='Output file path (default: processed_<input>)')
    process_parser.add_argument('--no-remove-bg', action='store_true', help='Do not remove background')
    process_parser.add_argument('--no-enhance', action='store_true', help='Do not enhance image quality')
    process_parser.add_argument('--no-background', action='store_true', help='Do not add professional background')
    process_parser.add_argument('--background-type', choices=['gradient', 'solid', 'blur'], 
                               default='gradient', help='Type of background to add')
    process_parser.add_argument('--bg-color', help='Background color (R,G,B format, e.g., "240,240,240")')
    process_parser.add_argument('--brightness', type=float, default=1.1, help='Brightness adjustment (default: 1.1)')
    process_parser.add_argument('--contrast', type=float, default=1.1, help='Contrast adjustment (default: 1.1)')
    process_parser.add_argument('--sharpness', type=float, default=1.2, help='Sharpness adjustment (default: 1.2)')
    process_parser.add_argument('--color', type=float, default=1.1, help='Color saturation adjustment (default: 1.1)')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate headshot from text prompt')
    generate_parser.add_argument('prompt', help='Text description of the desired headshot')
    generate_parser.add_argument('-o', '--output', default='generated_headshot.jpg', 
                                help='Output file path (default: generated_headshot.jpg)')
    generate_parser.add_argument('--negative-prompt', 
                                default='low quality, blurry, distorted, deformed, extra limbs, bad anatomy',
                                help='Negative prompt (what to avoid)')
    generate_parser.add_argument('--width', type=int, default=512, help='Output width (default: 512)')
    generate_parser.add_argument('--height', type=int, default=512, help='Output height (default: 512)')
    generate_parser.add_argument('--steps', type=int, default=50, help='Number of inference steps (default: 50)')
    generate_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (default: 7.5)')
    generate_parser.add_argument('--model', default='runwayml/stable-diffusion-v1-5',
                                help='Stable Diffusion model to use')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple photos in a directory')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('output_dir', help='Output directory for processed images')
    batch_parser.add_argument('--no-remove-bg', action='store_true', help='Do not remove background')
    batch_parser.add_argument('--no-enhance', action='store_true', help='Do not enhance image quality')
    batch_parser.add_argument('--no-background', action='store_true', help='Do not add professional background')
    batch_parser.add_argument('--background-type', choices=['gradient', 'solid', 'blur'], 
                               default='gradient', help='Type of background to add')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize the generator
    try:
        logger.info("Initializing headshot generator...")
        generator = HeadshotGenerator()
        logger.info("Generator initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        sys.exit(1)
    
    if args.command == 'process':
        process_single_image(generator, args)
    elif args.command == 'generate':
        generate_from_prompt(generator, args)
    elif args.command == 'batch':
        batch_process_images(generator, args)

def process_single_image(generator, args):
    """Process a single image."""
    input_path = args.input
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_file = Path(input_path)
        output_path = input_file.parent / f"processed_{input_file.name}"
    
    # Parse background color if provided
    bg_color = None
    if args.bg_color:
        try:
            bg_color = tuple(map(int, args.bg_color.split(',')))
            if len(bg_color) != 3:
                raise ValueError("Color must have 3 values (R,G,B)")
        except ValueError as e:
            logger.error(f"Invalid background color format: {e}")
            sys.exit(1)
    
    try:
        logger.info(f"Processing image: {input_path}")
        
        # Process the image
        processed_image = generator.process_existing_photo(
            input_path,
            remove_bg=not args.no_remove_bg,
            enhance=not args.no_enhance,
            add_background=not args.no_background,
            background_type=args.background_type
        )
        
        # Apply custom enhancements
        if not args.no_enhance:
            processed_image = generator.enhance_image(
                processed_image,
                brightness=args.brightness,
                contrast=args.contrast,
                sharpness=args.sharpness,
                color=args.color
            )
        
        # Add custom background if specified
        if not args.no_background and bg_color:
            processed_image = generator.add_professional_background(
                processed_image,
                background_type=args.background_type,
                color1=bg_color,
                color2=bg_color
            )
        
        # Save the result
        processed_image.save(output_path)
        logger.info(f"Professional headshot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

def generate_from_prompt(generator, args):
    """Generate headshot from text prompt."""
    try:
        logger.info("Loading Stable Diffusion model...")
        generator.load_stable_diffusion(args.model)
        
        logger.info(f"Generating headshot with prompt: {args.prompt}")
        
        generated_image = generator.generate_headshot_from_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )
        
        # Save the result
        generated_image.save(args.output)
        logger.info(f"Generated headshot saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

def batch_process_images(generator, args):
    """Process multiple images in batch."""
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    try:
        logger.info(f"Starting batch processing: {input_dir} -> {output_dir}")
        
        processed_files = generator.batch_process(
            input_dir=input_dir,
            output_dir=output_dir,
            remove_bg=not args.no_remove_bg,
            enhance=not args.no_enhance,
            add_background=not args.no_background,
            background_type=args.background_type
        )
        
        logger.info(f"Batch processing completed! Processed {len(processed_files)} images.")
        
        for file_path in processed_files:
            logger.info(f"  ✓ {file_path}")
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
