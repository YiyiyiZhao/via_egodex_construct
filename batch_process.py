"""
Batch process all video frames in EgoDex dataset
Generate trajectory.txt for each .frames directory
"""
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from datetime import datetime

from utils.hand_detector import HandDetector
from utils.object_detector import ObjectDetector
from utils.contact_matcher import match_hands_objects
from utils.trajectory_builder import TrajectoryBuilder
import cv2


def natural_sort_key(filename):
    """Sort files by numeric order"""
    return int(Path(filename).stem)


def load_frames_from_directory(frames_dir):
    """
    Load all frames from directory
    :param frames_dir: Directory containing frame images
    :return: List of (frame_idx, image, width, height)
    """
    frames_dir = Path(frames_dir)
    jpg_files = list(frames_dir.glob("*.jpg"))
    jpg_files.sort(key=lambda x: natural_sort_key(x.name))

    frames = []
    for jpg_file in jpg_files:
        img = cv2.imread(str(jpg_file))
        if img is None:
            print(f"Warning: Could not load {jpg_file}, skipping...")
            continue
        frame_idx = natural_sort_key(jpg_file.name)
        height, width = img.shape[:2]
        frames.append((frame_idx, img, width, height))

    return frames


def process_single_video(frames_dir, hand_detector, obj_detector, use_relative_direction=True,
                        obj_conf_threshold=0.3):
    """
    Process a single video directory
    :param frames_dir: Directory containing video frames
    :param hand_detector: HandDetector instance (shared across videos)
    :param obj_detector: ObjectDetector instance (shared across videos)
    :param use_relative_direction: If True, output relative directions
    :param obj_conf_threshold: Object detection confidence threshold
    :return: (output_text, llm_format, success)
    """
    try:
        # Load frames
        frames = load_frames_from_directory(frames_dir)

        if len(frames) == 0:
            print(f"  ⚠ No frames found in {frames_dir}")
            return None, None, False

        # Initialize trajectory builder
        trajectory_builder = TrajectoryBuilder(use_relative_direction=use_relative_direction)

        # Process each frame
        for frame_idx, img, width, height in frames:
            # Detect hands
            hands = hand_detector.detect(img)

            # Detect objects
            objects = obj_detector.detect(img, conf_thres=obj_conf_threshold)

            # Match hands to objects
            match_result = match_hands_objects(hands, objects)

            # Add to trajectory
            trajectory_builder.add_frame(match_result)

        # Format output
        output = trajectory_builder.format_all_trajectories()
        llm_format = trajectory_builder.format_for_llm()

        return output, llm_format, True

    except Exception as e:
        print(f"  ❌ Error processing {frames_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def batch_process(frames_root_dir, hand_model_path, output_dir="./trajectories",
                 use_relative_direction=True, yolo_model="yolov8l-worldv2.pt",
                 obj_conf_threshold=0.3, start_index=0, limit=None):
    """
    Batch process all .frames directories

    :param frames_root_dir: Root directory containing all .frames subdirectories
    :param hand_model_path: Path to hand detector model
    :param output_dir: Output directory to save all trajectory files
    :param use_relative_direction: If True, output relative directions
    :param yolo_model: YOLO-World model name
    :param obj_conf_threshold: Object detection confidence threshold
    :param start_index: Start from this index (for resume)
    :param limit: Process only this many videos (None = all)
    """
    print("=" * 70)
    print("Batch Processing EgoDex Videos")
    print("=" * 70)
    print(f"Root directory: {frames_root_dir}")
    print(f"Hand model: {hand_model_path}")
    print(f"YOLO model: {yolo_model}")
    print(f"Object confidence threshold: {obj_conf_threshold}")
    print(f"Relative direction: {use_relative_direction}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory created: {output_path.absolute()}")

    # Find all .frames directories
    frames_root = Path(frames_root_dir)
    all_frame_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir() and d.name.endswith('.frames')])

    print(f"\nFound {len(all_frame_dirs)} .frames directories")

    # Apply start_index and limit
    if start_index > 0:
        all_frame_dirs = all_frame_dirs[start_index:]
        print(f"Starting from index {start_index}")

    if limit is not None:
        all_frame_dirs = all_frame_dirs[:limit]
        print(f"Processing only {limit} videos")

    print(f"Will process {len(all_frame_dirs)} videos\n")

    # Initialize detectors once (shared across all videos)
    print("Initializing detectors...")
    hand_detector = HandDetector(hand_model_path)
    obj_detector = ObjectDetector(model_name=yolo_model)
    print("✓ Detectors initialized\n")

    # Statistics
    total = len(all_frame_dirs)
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    # Process each directory
    for i, frames_dir in enumerate(all_frame_dirs, 1):
        dir_name = frames_dir.name
        # Save to output directory with video name
        output_filename = f"{dir_name}.txt"
        output_file = output_path / output_filename

        # Check if already processed
        if output_file.exists():
            print(f"[{i}/{total}] ⏭  {dir_name} - Already processed, skipping")
            skipped += 1
            continue

        print(f"[{i}/{total}] 🔄 Processing {dir_name}...")

        # Process video
        output, llm_format, success = process_single_video(
            frames_dir, hand_detector, obj_detector,
            use_relative_direction, obj_conf_threshold
        )

        if success:
            # Save to file
            with open(output_file, 'w') as f:
                f.write(output)
                f.write("\n\n## LLM-Friendly Format:\n")
                f.write(llm_format)

            print(f"[{i}/{total}] ✓ {dir_name} - Saved to {output_filename}")
            successful += 1
        else:
            print(f"[{i}/{total}] ✗ {dir_name} - Failed")
            failed += 1

        # Progress estimate
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / i
            remaining = (total - i) * avg_time
            eta = datetime.fromtimestamp(time.time() + remaining).strftime('%H:%M:%S')
            print(f"  Progress: {i}/{total} | Elapsed: {elapsed:.1f}s | ETA: {eta}\n")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    if successful > 0:
        print(f"Average time per video: {total_time/successful:.1f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Batch process EgoDex video frames')

    parser.add_argument('--frames_dir', type=str,
                       default="/data1/zy/full_data/EgoDex/frames",
                       help='Root directory containing .frames subdirectories')

    parser.add_argument('--hand_model', type=str,
                       default="/data1/zy/models/hand_detector/hand_detector.pth",
                       help='Path to hand detector model')

    parser.add_argument('--yolo_model', type=str,
                       default="yolov8l-worldv2.pt",
                       choices=["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt"],
                       help='YOLO-World model')

    parser.add_argument('--conf_threshold', type=float, default=0.3,
                       help='Object detection confidence threshold')

    parser.add_argument('--output_dir', type=str, default="./trajectories",
                       help='Output directory to save all trajectory files')

    parser.add_argument('--relative_direction', action='store_true', default=True,
                       help='Use relative directions instead of coordinates')

    parser.add_argument('--start_index', type=int, default=0,
                       help='Start from this index (for resuming)')

    parser.add_argument('--limit', type=int, default=None,
                       help='Process only N videos (for testing)')

    args = parser.parse_args()

    batch_process(
        frames_root_dir=args.frames_dir,
        hand_model_path=args.hand_model,
        output_dir=args.output_dir,
        use_relative_direction=args.relative_direction,
        yolo_model=args.yolo_model,
        obj_conf_threshold=args.conf_threshold,
        start_index=args.start_index,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
