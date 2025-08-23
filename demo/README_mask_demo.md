# Top-Down Video Demo with Mask

This script (`top_down_video_demo_with_mask.py`) is an alternative to the original `top_down_video_demo_with_mmdet.py` that uses a mask video to define regions of interest instead of human detection bounding boxes.

## Key Differences from Original Script

- **No MMDetection Required**: Does not require mmdet installation or detection models
- **Mask-Based Region Detection**: Uses a binary mask video to identify regions for pose estimation
- **Synchronized Video Processing**: Processes main video and mask video frame-by-frame

## Usage

```bash
python demo/top_down_video_demo_with_mask.py \
    <pose_config> \
    <pose_checkpoint> \
    --video-path <main_video> \
    --mask-video-path <mask_video> \
    [optional arguments]
```

### Required Arguments

- `pose_config`: Path to pose estimation config file (e.g., configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_*.py)
- `pose_checkpoint`: Path to pose estimation checkpoint file (e.g., models/vitpose_*.pth)
- `--video-path`: Path to the main video file
- `--mask-video-path`: Path to the binary mask video file

### Optional Arguments

- `--show`: Display the visualization in real-time
- `--out-video-root`: Directory to save output video (default: no saving)
- `--device`: Device for inference (default: 'cuda:0')
- `--min-area`: Minimum area for mask regions (default: 500 pixels)
- `--bbox-thr`: Legacy parameter for compatibility (default: 0.3)
- `--kpt-thr`: Keypoint confidence threshold (default: 0.3)
- `--radius`: Keypoint radius for visualization (default: 4)
- `--thickness`: Link thickness for visualization (default: 1)

## Mask Video Requirements

The mask video should be:

1. **Same frame count** as the main video
2. **Binary or grayscale**: White/bright regions indicate areas for pose estimation
3. **Same dimensions** as the main video (recommended)
4. **Synchronized**: Frame N of mask video corresponds to frame N of main video

### Creating Mask Videos

You can create mask videos using various methods:

1. **Manual annotation**: Use video editing software to create binary masks
2. **Object segmentation**: Use tools like SAM (Segment Anything Model) or other segmentation models
3. **Motion detection**: Create masks based on motion or optical flow
4. **ROI definition**: Define static regions of interest

## Example Usage

```bash
# Basic usage with display
python demo/top_down_video_demo_with_mask.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py \
    models/vitpose_base.pth \
    --video-path input_video.mp4 \
    --mask-video-path mask_video.mp4 \
    --show

# Save output video
python demo/top_down_video_demo_with_mask.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py \
    models/vitpose_base.pth \
    --video-path input_video.mp4 \
    --mask-video-path mask_video.mp4 \
    --out-video-root output/

# Adjust parameters
python demo/top_down_video_demo_with_mask.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py \
    models/vitpose_base.pth \
    --video-path input_video.mp4 \
    --mask-video-path mask_video.mp4 \
    --min-area 1000 \
    --kpt-thr 0.5 \
    --radius 6 \
    --thickness 2 \
    --show
```

## How It Works

1. **Frame Synchronization**: Reads corresponding frames from both videos
2. **Mask Processing**: Converts mask frame to binary and finds contours
3. **Bounding Box Extraction**: Creates bounding boxes around white regions in mask
4. **Pose Estimation**: Applies ViTPose to extracted regions
5. **Visualization**: Renders pose results on original video frames

## Advantages

- **Flexible Region Definition**: Use any method to create masks (manual, AI-generated, etc.)
- **No Detection Model Needed**: Eliminates dependency on human detection models
- **Custom ROI**: Define exactly where to look for poses
- **Multi-object Support**: Can handle multiple regions per frame

## Use Cases

- **Sports Analysis**: Define playing field areas or specific zones
- **Surveillance**: Focus on specific areas of interest
- **Performance Analysis**: Track poses in predefined regions
- **Custom Scenarios**: Any situation where you want precise control over analysis regions
