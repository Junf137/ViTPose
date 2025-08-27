#!/bin/bash

video_root="/root/Videos/challenging_videos"
mask_root="/root/Videos/sam2_carego_mask"
output_root="/root/Videos/output/vitpose_huge_body_coco"
mask_bbox_video_root="root/Videos/output/vitpose_huge_body_coco"
bbox_output_root="output/bbox"
pose_output_root="output/pose"

for video_path in $video_root/*.mp4; do
    video_name=$(basename $video_path)
    mask_path="$mask_root/${video_name%.mp4}_carego_sam2_masks_only.mp4"

    echo "Processing $video_name"
    echo "Mask path: $mask_path"
    echo "Output path: $output_root"

    python demo/top_down_video_demo_with_mask.py \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
        models/vitpose-h-multi-coco.pth \
        --video-path $video_path \
        --mask-video-path $mask_path \
        --out-video-root $output_root \
        --bbox-output-root $bbox_output_root \
        --pose-output-root $pose_output_root \
        --mask-bbox-video-root $mask_bbox_video_root
done
