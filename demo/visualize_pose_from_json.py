# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import numpy as np
import cv2
from argparse import ArgumentParser


# COCO 17 keypoint names for reference
COCO_KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]


def load_pose_data(json_path):
    """Load pose data from JSON file.

    Args:
        json_path: Path to pose JSON file

    Returns:
        Dictionary with frame-wise pose data
    """
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    # Convert string keys to integers for easier processing
    converted_data = {}
    for frame_str, poses in pose_data.items():
        converted_data[int(frame_str)] = poses

    return converted_data


def draw_keypoints_with_indices(img, keypoints, kpt_thr=0.3, radius=4,
                               font_scale=0.4, font_thickness=1, draw_lines=True):
    """Draw keypoints with index labels and connecting lines on image.

    Args:
        img: Image to draw on (BGR format)
        keypoints: Array of keypoints in format [[x, y, conf], ...]
        kpt_thr: Confidence threshold for showing keypoints
        radius: Radius of keypoint circles
        font_scale: Font scale for index labels
        font_thickness: Font thickness for index labels
        draw_lines: Whether to draw connecting lines between arm joints

    Returns:
        Modified image with keypoints, indices and connecting lines drawn
    """
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)

    # Colors for keypoints (different colors for better visibility)
    colors = [
        (255, 0, 0),    # nose - red
        (255, 85, 0),   # left_eye - orange
        (255, 170, 0),  # right_eye - yellow-orange
        (255, 255, 0),  # left_ear - yellow
        (170, 255, 0),  # right_ear - yellow-green
        (85, 255, 0),   # left_shoulder - light green
        (0, 255, 0),    # right_shoulder - green
        (0, 255, 85),   # left_elbow - green-cyan
        (0, 255, 170),  # right_elbow - cyan-green
        (0, 255, 255),  # left_wrist - cyan
        (0, 170, 255),  # right_wrist - light blue
        (0, 85, 255),   # left_hip - blue
        (0, 0, 255),    # right_hip - dark blue
        (85, 0, 255),   # left_knee - purple-blue
        (170, 0, 255),  # right_knee - purple
        (255, 0, 255),  # left_ankle - magenta
        (255, 0, 170)   # right_ankle - pink
    ]

    # Define connections for drawing lines
    # Format: (start_kpt_idx, end_kpt_idx, line_color, line_thickness)
    connections = [
        # Left arm: wrist -> elbow -> shoulder
        (9, 7, (0, 255, 255), 3),    # left_wrist -> left_elbow (cyan)
        (7, 5, (0, 255, 255), 3),    # left_elbow -> left_shoulder (cyan)

        # Right arm: wrist -> elbow -> shoulder
        (10, 8, (0, 170, 255), 3),   # right_wrist -> right_elbow (light blue)
        (8, 6, (0, 170, 255), 3),    # right_elbow -> right_shoulder (light blue)
    ]

    # Draw connecting lines first (so they appear behind keypoints)
    if draw_lines:
        for start_idx, end_idx, line_color, line_thickness in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                start_idx in [5, 6, 7, 8, 9, 10] and end_idx in [5, 6, 7, 8, 9, 10]):

                start_kpt = keypoints[start_idx]
                end_kpt = keypoints[end_idx]

                start_x, start_y, start_conf = float(start_kpt[0]), float(start_kpt[1]), float(start_kpt[2])
                end_x, end_y, end_conf = float(end_kpt[0]), float(end_kpt[1]), float(end_kpt[2])

                # Only draw line if both keypoints are above confidence threshold
                if start_conf > kpt_thr and end_conf > kpt_thr:
                    start_point = (int(start_x), int(start_y))
                    end_point = (int(end_x), int(end_y))

                    # Draw line with black border for better visibility
                    cv2.line(img, start_point, end_point, (0, 0, 0), line_thickness + 2)
                    cv2.line(img, start_point, end_point, line_color, line_thickness)

    # Draw keypoints and labels
    for kpt_idx, kpt in enumerate(keypoints):
        # showing only the [5, 6, 7, 8, 9, 10] keypoints
        if kpt_idx not in [5, 6, 7, 8, 9, 10]:
            continue

        x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])

        # Only draw if confidence is above threshold
        if conf > kpt_thr:
            x_int, y_int = int(x), int(y)
            color = colors[kpt_idx % len(colors)]

            # Draw keypoint circle
            cv2.circle(img, (x_int, y_int), radius, color, -1)
            cv2.circle(img, (x_int, y_int), radius + 1, (0, 0, 0), 1)  # Black border

            # Draw index label
            label = str(kpt_idx)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thickness)[0]

            # Position label to avoid overlapping with keypoint
            label_x = x_int + radius + 5
            label_y = y_int - radius - 5

            # Keep label within image bounds
            if label_x + label_size[0] >= img.shape[1]:
                label_x = x_int - label_size[0] - radius - 5
            if label_y - label_size[1] <= 0:
                label_y = y_int + radius + label_size[1] + 5

            # Draw label background
            cv2.rectangle(img,
                         (label_x - 2, label_y - label_size[1] - 2),
                         (label_x + label_size[0] + 2, label_y + 2),
                         (255, 255, 255), -1)
            cv2.rectangle(img,
                         (label_x - 2, label_y - label_size[1] - 2),
                         (label_x + label_size[0] + 2, label_y + 2),
                         (0, 0, 0), 1)

            # Draw label text
            cv2.putText(img, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 0, 0), font_thickness)

    return img


def draw_bounding_boxes(img, bboxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image.

    Args:
        img: Image to draw on
        bboxes: List of bounding boxes in format [x1, y1, x2, y2, score]
        color: Color of bounding boxes in BGR format
        thickness: Thickness of bounding box lines

    Returns:
        Modified image with bounding boxes drawn
    """
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        score = bbox[4] if len(bbox) > 4 else 1.0

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label with person ID and score
        label = f"Person {i}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]

        # Position label above bounding box
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

        # Draw label background
        cv2.rectangle(img, (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0] + 5, label_y + 5),
                     color, -1)

        # Draw label text
        cv2.putText(img, label, (x1 + 2, label_y - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img


def main():
    """Visualize pose estimation results from JSON file on video."""
    parser = ArgumentParser()
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('pose_json', help='Path to pose JSON file')
    parser.add_argument('--output-video', type=str, required=True,
                       help='Path to output video file')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                       help='Keypoint confidence threshold')
    parser.add_argument('--radius', type=int, default=4,
                       help='Keypoint circle radius')
    parser.add_argument('--font-scale', type=float, default=0.4,
                       help='Font scale for keypoint indices')
    parser.add_argument('--bbox-thickness', type=int, default=2,
                       help='Bounding box line thickness')
    parser.add_argument('--show', action='store_true',
                       help='Show video during processing')
    parser.add_argument('--fps', type=float, default=None,
                       help='Output video FPS (default: same as input)')
    parser.add_argument('--no-lines', action='store_true',
                       help='Disable drawing connecting lines between arm joints')

    args = parser.parse_args()

    # Load pose data
    print(f"Loading pose data from {args.pose_json}...")
    pose_data = load_pose_data(args.pose_json)
    print(f"Loaded pose data for {len(pose_data)} frames")

    # Open input video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if args.fps is None else args.fps

    print(f"Video properties: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")

    # Setup output video writer
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    frame_idx = 0
    processed_frames = 0

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Get pose data for current frame
        frame_poses = pose_data.get(frame_idx, [])

        # Draw poses on frame
        vis_frame = frame.copy()

        if frame_poses:
            # Collect all bounding boxes for this frame
            bboxes = []

            for pose in frame_poses:
                # Draw keypoints with indices
                if 'keypoints' in pose:
                    keypoints = pose['keypoints']
                    vis_frame = draw_keypoints_with_indices(
                        vis_frame, keypoints,
                        kpt_thr=args.kpt_thr,
                        radius=args.radius,
                        font_scale=args.font_scale,
                        draw_lines=not args.no_lines
                    )

                # Collect bounding box
                if 'bbox' in pose:
                    bboxes.append(pose['bbox'])

            # Draw all bounding boxes
            if bboxes:
                vis_frame = draw_bounding_boxes(
                    vis_frame, bboxes,
                    thickness=args.bbox_thickness
                )

        # Add frame info
        info_text = f"Frame: {frame_idx}/{total_frames} | Persons: {len(frame_poses)}"
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Write frame to output video
        out.write(vis_frame)
        processed_frames += 1

        # Show frame if requested
        if args.show:
            cv2.imshow('Pose Visualization', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    # Cleanup
    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Video processing complete!")
    print(f"Processed {processed_frames} frames")
    print(f"Output saved to: {args.output_video}")


if __name__ == '__main__':
    main()
