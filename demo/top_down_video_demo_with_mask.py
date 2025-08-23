# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import numpy as np
from argparse import ArgumentParser

import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def extract_bounding_boxes_from_mask(mask_frame, min_area=500, bbox_thr=0.5):
    """Extract bounding boxes from mask frame.

    Args:
        mask_frame: Binary mask frame (grayscale)
        min_area: Minimum contour area to consider
        bbox_thr: Confidence threshold for bounding boxes (dummy value for compatibility)

    Returns:
        List of bounding boxes in format [x1, y1, x2, y2, score]
    """
    # Convert to grayscale if needed
    if len(mask_frame.shape) == 3:
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    # Threshold the mask to ensure binary
    _, binary_mask = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Format: [x1, y1, x2, y2, score]
            bboxes.append([x, y, x + w, y + h, 1.0])  # Score is 1.0 since mask is binary

    return bboxes


def main():
    """Visualize the demo videos using mask video for region detection.

    Using a mask video to define regions of interest instead of mmdet.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Main video path')
    parser.add_argument('--mask-video-path', type=str, help='Mask video path (binary mask)')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--min-area',
        type=int,
        default=500,
        help='Minimum area for mask regions to be considered')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold (not used for mask, kept for compatibility)')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.video_path is not None, 'Please provide --video-path'
    assert args.mask_video_path is not None, 'Please provide --mask-video-path'

    # Build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # Open main video
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    # Open mask video
    mask_cap = cv2.VideoCapture(args.mask_video_path)
    assert mask_cap.isOpened(), f'Failed to load mask video file {args.mask_video_path}'

    # Check that both videos have the same frame count
    main_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_frame_count = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Main video frames: {main_frame_count}, Mask video frames: {mask_frame_count}")

    if main_frame_count != mask_frame_count:
        warnings.warn(f'Frame count mismatch: main video has {main_frame_count} frames, '
                     f'mask video has {mask_frame_count} frames. Using minimum count.')

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # Optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    frame_idx = 0
    while (cap.isOpened() and mask_cap.isOpened()):
        flag, img = cap.read()
        mask_flag, mask_frame = mask_cap.read()

        if not flag or not mask_flag:
            break

        frame_idx += 1
        # print(f"Processing frame {frame_idx}")

        # Extract bounding boxes from mask
        mask_results = extract_bounding_boxes_from_mask(
            mask_frame, min_area=args.min_area, bbox_thr=args.bbox_thr)

        if len(mask_results) == 0:
            print(f"No valid regions found in mask frame {frame_idx}")
            # Create empty visualization
            vis_img = img.copy()
        else:
            print(f"Found {len(mask_results)} regions in mask frame {frame_idx}")

            # Convert mask results to the format expected by mmpose
            person_results = []
            for bbox in mask_results:
                person_results.append({
                    'bbox': np.array(bbox)  # [x1, y1, x2, y2, score]
                })

            # Test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            # Show the results
            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False)

        if args.show:
            cv2.imshow('Image', vis_img)
            cv2.imshow('Mask', mask_frame)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    mask_cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
