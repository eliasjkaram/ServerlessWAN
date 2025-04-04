import os
import argparse
import base64
import json
import random
import sys
import io
import time
import traceback
from typing import Dict, List, Any
import logging

import torch
import numpy as np
from PIL import Image
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
import cv2

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global model cache
MODEL_CACHE = {}

def setup_and_validate_args(job_input):
    try:
        args = argparse.Namespace()
        
        # Set default values
        args.task = job_input.get("task", "t2v-14B")
        args.size = job_input.get("size", "1280*720")
        args.frame_num = job_input.get("frame_num", 81)
        args.ckpt_dir = os.environ.get("MODEL_PATH", "/runpod-volume/wan-models")
        args.offload_model = job_input.get("offload_model", True)
        args.ulysses_size = 1
        args.ring_size = 1
        args.t5_fsdp = False
        args.t5_cpu = False
        args.dit_fsdp = False
        args.prompt = job_input.get("prompt", None)
        args.use_prompt_extend = job_input.get("use_prompt_extend", False)
        args.prompt_extend_method = job_input.get("prompt_extend_method", "local_qwen")
        args.prompt_extend_model = job_input.get("prompt_extend_model", None)
        args.prompt_extend_target_lang = job_input.get("prompt_extend_target_lang", "zh")
        args.base_seed = job_input.get("seed", -1)
        args.image = job_input.get("image", None)
        args.sample_solver = job_input.get("sample_solver", "unipc")
        args.sample_steps = job_input.get("sample_steps", None)
        args.sample_shift = job_input.get("sample_shift", None)
        args.sample_guide_scale = job_input.get("sample_guide_scale", 5.0)
        
        # Import modules here to avoid import errors before we're ready
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
        
        # Basic check
        if args.task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported task: {args.task}")
        
        # Check if model directory exists
        if not os.path.exists(args.ckpt_dir):
            raise ValueError(f"Model checkpoint directory not found: {args.ckpt_dir}")
        
        # Check if specific model directory exists
        task_dir = os.path.join(args.ckpt_dir, args.task)
        if not os.path.exists(task_dir):
            raise ValueError(f"Model directory for task {args.task} not found at {task_dir}")
        
        # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
        if args.sample_steps is None:
            args.sample_steps = 40 if "i2v" in args.task else 50

        if args.sample_shift is None:
            args.sample_shift = 5.0
            if "i2v" in args.task and args.size in ["832*480", "480*832"]:
                args.sample_shift = 3.0

        # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
        if args.frame_num is None:
            args.frame_num = 1 if "t2i" in args.task else 81

        # T2I frame_num check
        if "t2i" in args.task:
            if args.frame_num != 1:
                logger.warning(f"Adjusting frame_num from {args.frame_num} to 1 for task {args.task}")
                args.frame_num = 1

        args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
        
        # Size check
        if args.size not in SUPPORTED_SIZES[args.task]:
            supported = ", ".join(SUPPORTED_SIZES[args.task])
            raise ValueError(f"Unsupported size {args.size} for task {args.task}. Supported sizes: {supported}")
        
        # Check prompt or image based on task
        if "i2v" in args.task:
            if args.image is None:
                raise ValueError("Image is required for image-to-video task")
        else:
            if args.prompt is None:
                raise ValueError("Prompt is required for text-to-image/video task")
        
        return args
    except Exception as e:
        logger.error(f"Error setting up arguments: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_or_create_model(args):
    """Get model from cache or create a new one."""
    model_key = f"{args.task}"
    
    try:
        if model_key in MODEL_CACHE:
            logger.info(f"Using cached model for {model_key}")
            return MODEL_CACHE[model_key]
        
        logger.info(f"Creating new model for {model_key}")
        
        # Import wan locally to prevent import errors
        import wan
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError("CUDA is not available but required for model inference")
        
        # Log GPU info
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Using GPU: {device_name} (Device {current_device}, Total devices: {device_count})")
        
        # Check model files exist
        task_dir = os.path.join(args.ckpt_dir, args.task)
        if not os.path.exists(task_dir):
            raise FileNotFoundError(f"Model directory not found: {task_dir}")
        
        # List available files in model directory
        logger.info(f"Model directory contents: {os.listdir(task_dir)}")
        
        model = wan.get_model(
            task=args.task,
            ckpt_dir=args.ckpt_dir,
            offload_model=args.offload_model,
            t5_fsdp=args.t5_fsdp,
            t5_cpu=args.t5_cpu,
            dit_fsdp=args.dit_fsdp,
        )
        
        MODEL_CACHE[model_key] = model
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def process_image(image_url):
    """Download and process image from URL."""
    try:
        if image_url is None:
            raise ValueError("Image URL is None")
            
        if isinstance(image_url, bytes):
            # Handle bytes directly
            image = Image.open(io.BytesIO(image_url))
            return image
            
        if image_url.startswith("data:image"):
            # Handle base64 encoded image
            try:
                image_data = image_url.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                logger.error(f"Error processing base64 image: {str(e)}")
                raise ValueError(f"Invalid base64 image data: {str(e)}")
        else:
            # Handle URL
            try:
                image_path = rp_download(image_url)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Downloaded image not found at {image_path}")
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Error downloading image from URL: {str(e)}")
                raise ValueError(f"Failed to download or process image from URL: {str(e)}")
        
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def encode_video(video_frames, fps=30):
    """Encode video frames to base64."""
    try:
        if not video_frames or len(video_frames) == 0:
            raise ValueError("No video frames to encode")
            
        height, width, _ = video_frames[0].shape
        
        # Create a temporary file
        temp_file = "/tmp/output_video.mp4"
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
        
        # Write frames
        for frame in video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        # Check if file was created successfully
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            raise IOError(f"Failed to create video file at {temp_file}")
        
        # Read the file and encode to base64
        with open(temp_file, "rb") as f:
            video_bytes = f.read()
        
        # Remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding video: {str(e)}")
        logger.error(traceback.format_exc())
        if os.path.exists("/tmp/output_video.mp4"):
            os.remove("/tmp/output_video.mp4")
        raise

def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def handler(job):
    """Handler function for RunPod serverless."""
    job_id = job.get("id", "unknown")
    logger.info(f"Starting job {job_id}")
    
    try:
        # Extract input from job
        if not isinstance(job, dict):
            raise ValueError(f"Expected job to be a dictionary, got {type(job)}")
            
        if "input" not in job:
            raise ValueError("No 'input' field in job")
            
        job_input = job["input"]
        if not isinstance(job_input, dict):
            raise ValueError(f"Expected job['input'] to be a dictionary, got {type(job_input)}")
        
        # Log the input (excluding large fields like image)
        safe_input = {k: v for k, v in job_input.items() if k != "image"}
        logger.info(f"Job input: {json.dumps(safe_input)}")
        
        # Setup and validate arguments
        args = setup_and_validate_args(job_input)
        
        # Get or create model
        model = get_or_create_model(args)
        
        # Process inputs
        input_image = None
        if args.image:
            input_image = process_image(args.image)
            logger.info(f"Processed input image of size {input_image.size}")
        
        # Generate
        start_time = time.time()
        logger.info(f"Starting generation with task: {args.task}")
        
        try:
            with torch.no_grad():
                # Generate video or image
                if "i2v" in args.task:
                    logger.info("Running image-to-video generation")
                    output = model.image_to_video(
                        image=input_image,
                        size=args.size,
                        num_frames=args.frame_num,
                        seed=args.base_seed,
                        step=args.sample_steps,
                        solver=args.sample_solver,
                        shift=args.sample_shift,
                        cfg_scale=args.sample_guide_scale,
                    )
                elif "t2i" in args.task:
                    logger.info("Running text-to-image generation")
                    output = model.text_to_image(
                        prompt=args.prompt,
                        size=args.size,
                        seed=args.base_seed,
                        step=args.sample_steps,
                        solver=args.sample_solver,
                        shift=args.sample_shift,
                        cfg_scale=args.sample_guide_scale,
                    )
                else:  # t2v
                    logger.info("Running text-to-video generation")
                    output = model.text_to_video(
                        prompt=args.prompt,
                        size=args.size,
                        num_frames=args.frame_num,
                        seed=args.base_seed,
                        step=args.sample_steps,
                        solver=args.sample_solver,
                        shift=args.sample_shift,
                        cfg_scale=args.sample_guide_scale,
                    )
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Generation failed: {str(e)}")
        finally:
            # Clear CUDA cache regardless of success/failure
            clear_cuda_cache()
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        
        # Process output
        try:
            if "t2i" in args.task:
                # Convert PIL image to base64
                logger.info("Processing text-to-image output")
                buffer = io.BytesIO()
                output.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                output_data = {
                    "image": img_str,
                    "seed": args.base_seed,
                    "format": "PNG"
                }
            else:
                # Handle video output
                logger.info("Processing video output")
                if hasattr(output, "frames"):
                    frames = output.frames
                else:
                    frames = [np.array(frame) for frame in output]
                
                logger.info(f"Encoding {len(frames)} video frames")
                video_b64 = encode_video(frames)
                output_data = {
                    "video": video_b64,
                    "seed": args.base_seed,
                    "frame_count": len(frames),
                    "format": "MP4"
                }
        except Exception as e:
            logger.error(f"Error processing output: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to process output: {str(e)}")
        
        # Create final response
        response = {
            "output": output_data,
            "stats": {
                "generation_time": generation_time,
                "task": args.task,
                "size": args.size,
                "sample_steps": args.sample_steps
            }
        }
        
        logger.info(f"Job {job_id} completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        # Clean up any temporary files
        rp_cleanup()
        # Clear CUDA cache again to ensure memory is freed
        clear_cuda_cache()

# Start the serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler}) 