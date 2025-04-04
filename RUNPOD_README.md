# WAN 2.1 RunPod Serverless Deployment

This repository contains the necessary files to deploy WAN 2.1 (Video Generation) as a RunPod serverless endpoint.

## Prerequisites

1. A RunPod account with access to serverless features
2. Access to the WAN 2.1 model checkpoints 
3. Docker installed locally if you want to build the image yourself

## Setup Instructions

### Option 1: Deploy Using Pre-built Image

1. Log in to your RunPod account and navigate to the Serverless section
2. Create a new endpoint using the Docker image: `YOUR_DOCKERHUB_USERNAME/wan-video-generator:2.1.0`
3. Attach a volume to the endpoint that contains the WAN model checkpoints
4. Make sure the volume is mounted at `/runpod-volume/wan-models`
5. Select the appropriate GPU (minimum 24GB VRAM)
6. Set the worker count according to your needs
7. Create the endpoint

### Option 2: Build and Deploy Your Own Image

1. Clone this repository and navigate to its directory
2. Build the Docker image:
   ```bash
   docker build -t wan-video-generator:2.1.0 .
   ```
3. Tag and push the image to your Docker registry:
   ```bash
   docker tag wan-video-generator:2.1.0 YOUR_DOCKERHUB_USERNAME/wan-video-generator:2.1.0
   docker push YOUR_DOCKERHUB_USERNAME/wan-video-generator:2.1.0
   ```
4. Follow steps 1-7 from Option 1, using your custom image URL

## Model Checkpoint Setup

The WAN 2.1 model requires specific checkpoint files to function. Make sure these files are available in your attached volume at `/runpod-volume/wan-models`.

The checkpoint directory structure should be:
```
/runpod-volume/wan-models/
├── t2v-14B/
│   ├── dit.pt
│   └── t5.pt
├── t2v-1.3B/
│   ├── dit.pt
│   └── t5.pt
├── t2i-14B/
│   ├── dit.pt
│   └── t5.pt
└── i2v-14B/
    └── dit.pt
```

## Using the API

Once deployed, you can access your endpoint via the RunPod API. Here are some example requests:

### Text-to-Video Generation

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "task": "t2v-14B",
      "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
      "size": "1280*720",
      "frame_num": 81,
      "seed": 42,
      "sample_steps": 50,
      "sample_solver": "unipc",
      "sample_guide_scale": 5.0
    }
  }'
```

### Image-to-Video Generation

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "task": "i2v-14B",
      "image": "https://example.com/your-image.jpg",
      "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
      "size": "1280*720",
      "frame_num": 81,
      "seed": 42,
      "sample_steps": 40
    }
  }'
```

### Text-to-Image Generation

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "task": "t2i-14B",
      "prompt": "一个朴素端庄的美人",
      "size": "1024*1024",
      "seed": 42,
      "sample_steps": 50
    }
  }'
```

## API Parameters

See the `runpod.yaml` file for a full list of input and output parameters.

## Troubleshooting

1. **Out of Memory Errors**: Try setting `offload_model` to `true` and/or use a GPU with more VRAM
2. **Slow Generation**: Decrease `sample_steps` (minimum 20 recommended)
3. **Model Loading Errors**: Check that your checkpoint files are correctly placed in the volume
4. **API Timeout**: For longer generations, use the async endpoint and increase your RunPod timeout settings 