import os
import json
import base64
from PIL import Image
import io
import argparse

# Mock the runpod module
class MockRunPod:
    @staticmethod
    def serverless_start(config):
        print("Mock RunPod serverless start called with:", config)

class MockJob:
    def __init__(self, input_data):
        self.input_data = input_data
        
    @property
    def id(self):
        return "test-job-123"
        
    @property
    def input(self):
        return self.input_data

# Save the test image
def decode_and_save_output(output, task):
    if "error" in output:
        print(f"Error: {output['error']}")
        return
    
    if task.startswith("t2i"):
        # Save the image
        img_data = base64.b64decode(output["output"]["image"])
        with open("test_output.png", "wb") as f:
            f.write(img_data)
        print(f"Saved test output to test_output.png")
    else:
        # Save the video
        video_data = base64.b64decode(output["output"]["video"])
        with open("test_output.mp4", "wb") as f:
            f.write(video_data)
        print(f"Saved test output to test_output.mp4")
    
    # Print the stats
    print("Generation stats:")
    print(json.dumps(output["stats"], indent=2))

def get_test_image_base64(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    return f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"

def parse_args():
    parser = argparse.ArgumentParser(description="Test the WAN handler")
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        choices=["t2v-14B", "t2v-1.3B", "t2i-14B", "i2v-14B"],
        help="The task to test"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        help="The prompt to use for text-to-image/video tasks"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/i2v_input.JPG",
        help="Path to the image to use for image-to-video task"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of sampling steps (lower for faster testing)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=25,
        help="Number of frames to generate (for video tasks)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Mock runpod module
    import sys
    sys.modules["runpod"] = MockRunPod()
    sys.modules["runpod.serverless"] = MockRunPod()
    sys.modules["runpod.serverless.utils"] = type("MockRunPodUtils", (), {
        "rp_download": lambda url: url,
        "rp_upload": lambda file: file,
        "rp_cleanup": lambda: None
    })
    
    # Import the handler
    from handler import handler
    
    # Create input data
    if args.task.startswith("i2v"):
        # Image-to-video task
        input_data = {
            "task": args.task,
            "image": get_test_image_base64(args.image),
            "prompt": args.prompt,
            "size": "832*480",
            "frame_num": args.frames,
            "sample_steps": args.steps,
            "sample_solver": "unipc",
            "sample_guide_scale": 5.0
        }
    elif args.task.startswith("t2i"):
        # Text-to-image task
        input_data = {
            "task": args.task,
            "prompt": args.prompt,
            "size": "1024*1024",
            "sample_steps": args.steps,
            "sample_solver": "unipc",
            "sample_guide_scale": 5.0
        }
    else:
        # Text-to-video task
        input_data = {
            "task": args.task,
            "prompt": args.prompt,
            "size": "832*480",
            "frame_num": args.frames,
            "sample_steps": args.steps,
            "sample_solver": "unipc",
            "sample_guide_scale": 5.0
        }
    
    # Create the job object
    job = {"id": "test-job-123", "input": input_data}
    
    # Run the handler
    print(f"Testing handler with task: {args.task}")
    print(f"Input data: {json.dumps(input_data, indent=2)}")
    print("\nRunning handler, this may take a while...")
    
    output = handler(job)
    
    # Save the output
    decode_and_save_output(output, args.task)
    
if __name__ == "__main__":
    main() 