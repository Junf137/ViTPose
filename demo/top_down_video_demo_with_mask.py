# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import numpy as np
import json
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


def save_bbox_data(bbox_data, output_path):
    """Save bounding box data to JSON file.

    Args:
        bbox_data: Dictionary containing frame-wise bounding box data
        output_path: Path to output JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for frame_idx, bboxes in bbox_data.items():
        serializable_data[str(frame_idx)] = []
        for bbox in bboxes:
            if isinstance(bbox, np.ndarray):
                serializable_data[str(frame_idx)].append(bbox.tolist())
            else:
                serializable_data[str(frame_idx)].append(bbox)

    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"Bounding box data saved to: {output_path}")


def save_pose_data(pose_data, output_path):
    """Save pose results data to JSON file.

    Args:
        pose_data: Dictionary containing frame-wise pose results
        output_path: Path to output JSON file
    """
    # Convert pose results to serializable format
    serializable_data = {}
    for frame_idx, poses in pose_data.items():
        serializable_data[str(frame_idx)] = []
        for pose in poses:
            pose_dict = {}
            if 'bbox' in pose:
                pose_dict['bbox'] = pose['bbox'].tolist() if isinstance(pose['bbox'], np.ndarray) else pose['bbox']
            if 'keypoints' in pose:
                pose_dict['keypoints'] = pose['keypoints'].tolist() if isinstance(pose['keypoints'], np.ndarray) else pose['keypoints']
            # Add other pose attributes if they exist
            for key, value in pose.items():
                if key not in ['bbox', 'keypoints']:
                    if isinstance(value, np.ndarray):
                        pose_dict[key] = value.tolist()
                    else:
                        pose_dict[key] = value
            serializable_data[str(frame_idx)].append(pose_dict)

    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"Pose results data saved to: {output_path}")


def get_output_filename(video_path, suffix):
    """Get output filename based on input video name.

    Args:
        video_path: Path to input video
        suffix: Suffix to add to filename (e.g., 'bbox', 'pose')

    Returns:
        Output filename with suffix
    """
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    return f"{name_without_ext}_{suffix}.json"


def overlay_mask_on_image(img, mask_frame, alpha=0.5):
    """Overlay mask on the original image.

    Args:
        img: Original image (BGR format)
        mask_frame: Binary mask frame
        alpha: Transparency factor for mask overlay (0.0 to 1.0)

    Returns:
        Image with mask overlay
    """
    # Convert mask to 3-channel if needed
    if len(mask_frame.shape) == 3:
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_frame.copy()

    # Create colored mask (green overlay)
    mask_colored = np.zeros_like(img)
    mask_colored[:, :, 1] = mask_gray  # Green channel

    # Create binary mask for blending
    mask_binary = (mask_gray > 127).astype(np.float32)
    mask_3channel = np.stack([mask_binary] * 3, axis=2)

    # Blend the images
    result = img.astype(np.float32) * (1 - alpha * mask_3channel) + mask_colored.astype(np.float32) * alpha * mask_3channel

    return result.astype(np.uint8)


def draw_mask_bboxes(img, bboxes, color=(0, 255, 0), thickness=2):
    """Draw mask bounding boxes on image.

    Args:
        img: Image to draw on (will be modified in place)
        bboxes: List of bounding boxes in format [x1, y1, x2, y2, score]
        color: Color of bounding box in BGR format
        thickness: Thickness of bounding box lines

    Returns:
        Modified image with bounding boxes drawn
    """
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Add label
        label = f"Mask"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img


def add_keypoint_indices(img, pose_results, kpt_score_thr=0.3, font_scale=0.4, font_color=(255, 255, 255), font_thickness=1):
    """Add keypoint indices next to each keypoint in the image.

    Args:
        img: The image to draw on (will be modified in place)
        pose_results: List of pose results containing keypoints
        kpt_score_thr: Minimum keypoint score threshold to display index
        font_scale: Font scale for the text
        font_color: Color of the text in BGR format
        font_thickness: Thickness of the text

    Returns:
        Modified image with keypoint indices added
    """
    if not pose_results:
        return img

    for pose_result in pose_results:
        if 'keypoints' not in pose_result:
            continue

        keypoints = pose_result['keypoints']
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)

        # keypoints shape should be (num_joints, 3) where 3 = [x, y, visibility_score]
        for kpt_idx, kpt in enumerate(keypoints):
            x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

            # Only draw index if keypoint score is above threshold
            if kpt_score > kpt_score_thr:
                # Offset the text slightly from the keypoint to avoid overlap
                text_x = x_coord + 8
                text_y = y_coord - 8

                # Make sure text stays within image bounds
                text_y = max(15, text_y)  # Keep text at least 15 pixels from top
                text_x = min(img.shape[1] - 20, text_x)  # Keep text at least 20 pixels from right edge

                # Draw the keypoint index
                cv2.putText(img, str(kpt_idx), (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    return img


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
    parser.add_argument(
        '--bbox-output-root',
        default='',
        help='Root directory to save bounding box data. Default: no saving.')
    parser.add_argument(
        '--pose-output-root',
        default='',
        help='Root directory to save pose results data. Default: no saving.')
    parser.add_argument(
        '--mask-bbox-video-root',
        default='',
        help='Root directory to save mask+bbox overlay video. Default: no saving.')

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

    # Setup saving for bounding box and pose data
    save_bbox_data_flag = args.bbox_output_root != ''
    save_pose_data_flag = args.pose_output_root != ''

    if save_bbox_data_flag:
        os.makedirs(args.bbox_output_root, exist_ok=True)
        bbox_data = {}  # Dictionary to store frame-wise bounding box data

    if save_pose_data_flag:
        os.makedirs(args.pose_output_root, exist_ok=True)
        pose_data = {}  # Dictionary to store frame-wise pose results

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # Setup mask+bbox video saving
    save_mask_bbox_video = args.mask_bbox_video_root != ''
    if save_mask_bbox_video:
        os.makedirs(args.mask_bbox_video_root, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mask_bbox_videoWriter = cv2.VideoWriter(
            os.path.join(args.mask_bbox_video_root,
                         f'mask_bbox_{os.path.basename(args.video_path)}'), fourcc,
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

        # Save bounding box data if requested
        if save_bbox_data_flag:
            bbox_data[frame_idx] = mask_results

        # Create mask+bbox overlay video frame
        if save_mask_bbox_video:
            # Start with original image
            mask_bbox_img = img.copy()

            # Overlay mask on the image (green tint)
            mask_bbox_img = overlay_mask_on_image(mask_bbox_img, mask_frame, alpha=0.3)

            # Draw mask bounding boxes
            if len(mask_results) > 0:
                mask_bbox_img = draw_mask_bboxes(mask_bbox_img, mask_results)

        if len(mask_results) == 0:
            print(f"No valid regions found in mask frame {frame_idx}")
            # Create empty visualization
            vis_img = img.copy()
            # Save empty pose results if requested
            if save_pose_data_flag:
                pose_data[frame_idx] = []
        else:
            # print(f"Found {len(mask_results)} regions in mask frame {frame_idx}")

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

            # Save pose results if requested
            if save_pose_data_flag:
                pose_data[frame_idx] = pose_results

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

            # Add keypoint indices to the visualization
            vis_img = add_keypoint_indices(vis_img, pose_results, kpt_score_thr=args.kpt_thr)

        if args.show:
            cv2.imshow('Image', vis_img)
            cv2.imshow('Mask', mask_frame)
            if save_mask_bbox_video:
                cv2.imshow('Mask+BBox', mask_bbox_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if save_mask_bbox_video:
            mask_bbox_videoWriter.write(mask_bbox_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    mask_cap.release()
    if save_out_video:
        videoWriter.release()
    if save_mask_bbox_video:
        mask_bbox_videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()

    # Save collected data to files
    if save_bbox_data_flag and bbox_data:
        bbox_output_path = os.path.join(
            args.bbox_output_root,
            get_output_filename(args.video_path, 'bbox')
        )
        save_bbox_data(bbox_data, bbox_output_path)

    if save_pose_data_flag and pose_data:
        pose_output_path = os.path.join(
            args.pose_output_root,
            get_output_filename(args.video_path, 'pose')
        )
        save_pose_data(pose_data, pose_output_path)


if __name__ == '__main__':
    main()
