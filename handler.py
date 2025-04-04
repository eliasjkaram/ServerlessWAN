import os
import argparse
import base64
import json
import random
import sys
import io
import time
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

# Global model cache
MODEL_CACHE = {}

def setup_and_validate_args(job_input):
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
    
    # Basic check
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    
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
        assert args.frame_num == 1, f"Unsupported frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for task {args.task}, "
        f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )
    
    # Check prompt or image based on task
    if "i2v" in args.task:
        assert args.image is not None, "Image is required for image-to-video task"
    else:
        assert args.prompt is not None, "Prompt is required for text-to-image/video task"
    
    return args

def get_or_create_model(args):
    """Get model from cache or create a new one."""
    model_key = f"{args.task}"
    
    if model_key in MODEL_CACHE:
        logging.info(f"Using cached model for {model_key}")
        return MODEL_CACHE[model_key]
    
    logging.info(f"Creating new model for {model_key}")
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

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

def process_image(image_url):
    """Download and process image from URL."""
    if image_url.startswith("data:image"):
        # Handle base64 encoded image
        image_data = image_url.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
    else:
        # Handle URL
        image_path = rp_download(image_url)
        image = Image.open(image_path)
    
    return image

def encode_video(video_frames, fps=30):
    """Encode video frames to base64."""
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
    
    # Read the file and encode to base64
    with open(temp_file, "rb") as f:
        video_bytes = f.read()
    
    # Remove the temporary file
    os.remove(temp_file)
    
    return base64.b64encode(video_bytes).decode("utf-8")

def handler(job):
    """Handler function for RunPod serverless."""
    job_input = job["input"]
    
    try:
        # Setup and validate arguments
        args = setup_and_validate_args(job_input)
        
        # Get or create model
        model = get_or_create_model(args)
        
        # Process inputs
        if args.image:
            input_image = process_image(args.image)
        else:
            input_image = None
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            # Generate video or image
            if "i2v" in args.task:
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
        
        generation_time = time.time() - start_time
        logging.info(f"Generation completed in {generation_time:.2f} seconds")
        
        # Process output
        if "t2i" in args.task:
            # Convert PIL image to base64
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
            if hasattr(output, "frames"):
                frames = output.frames
            else:
                frames = [np.array(frame) for frame in output]
            
            video_b64 = encode_video(frames)
            output_data = {
                "video": video_b64,
                "seed": args.base_seed,
                "frame_count": len(frames),
                "format": "MP4"
            }
        
        return {
            "output": output_data,
            "stats": {
                "generation_time": generation_time,
                "task": args.task,
                "size": args.size,
                "sample_steps": args.sample_steps
            }
        }
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}

# Start the serverless handler
runpod.serverless.start({"handler": handler}) 