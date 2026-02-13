"""
Generate action descriptions using GPT vision model
Simple flow: Read frames -> Build prompt -> Call GPT -> Save results
"""
import os
import json
import base64
from pathlib import Path
from openai import OpenAI

from prompts import construct_examples, construct_task_prompt, get_action_label_from_dir


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def read_frames(frames_dir):
    """
    Read all images from frames directory (0.jpg, 1.jpg, ..., 8.jpg or 10.jpg)
    Sort by numerical order
    """
    frames_dir = Path(frames_dir)
    jpg_files = list(frames_dir.glob("*.jpg"))
    jpg_files.sort(key=lambda x: int(x.stem))  # Sort by number

    images = []
    for jpg_file in jpg_files:
        base64_img = encode_image(jpg_file)
        images.append(base64_img)

    print(f"  Read {len(images)} images: {[f.name for f in jpg_files]}")
    return images


def read_trajectory(trajectory_file):
    """Read trajectory file LLM format section"""
    trajectory_file = Path(trajectory_file)
    if not trajectory_file.exists():
        return "No trajectory information"

    with open(trajectory_file, 'r') as f:
        content = f.read()

    if "## LLM-Friendly Format:" in content:
        return content.split("## LLM-Friendly Format:")[1].strip()
    return content.strip()


def call_gpt_vision(client, images_base64, prompt, model="gpt-5.2-2025-12-11"):
    """Call GPT vision API to generate description"""
    content = []

    # Add all images
    for img in images_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    # Call API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def process_one_video(client, frames_dir, trajectory_file, model="gpt-5.2-2025-12-11"):
    """Process one video: Read images -> Read trajectory -> Generate description"""
    images = read_frames(frames_dir)
    if not images:
        return None, False

    action_label = get_action_label_from_dir(frames_dir)
    trajectory = read_trajectory(trajectory_file)

    examples = construct_examples()
    prompt = construct_task_prompt(examples, action_label, trajectory)

    description = call_gpt_vision(client, images, prompt, model)
    return description, True


def main():
    """Main function - all configuration here"""

    # ========== Configuration (change these if needed) ==========
    FRAMES_DIR = "/data1/zy/full_data/EgoDex/frames"
    TRAJECTORY_DIR = "./trajectories"
    OUTPUT_DIR = "./descriptions"
    MODEL = "gpt-5.2-2025-12-11"
    BASE_URL = "https://xiaoai.plus/v1"
    # =============================================================

    print("=" * 70)
    print("Batch Generate Action Descriptions")
    print("=" * 70)
    print(f"Frames: {FRAMES_DIR}")
    print(f"Trajectories: {TRAJECTORY_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print("=" * 70)

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client
    api_key = "sk-dm276hixX4n22oyP2aEc89915b9d4e1dA6E93e37A28850A7"
    client = OpenAI(api_key=api_key, base_url=BASE_URL)


    # Find all .frames directories
    frames_root = Path(FRAMES_DIR)
    all_dirs = sorted([d for d in frames_root.iterdir()
                      if d.is_dir() and d.name.endswith('.frames')])
    print(f"Found {len(all_dirs)} video directories\n")

    trajectory_path = Path(TRAJECTORY_DIR)
    results = []

    # Process each video
    for i, video_dir in enumerate(all_dirs, 1):
        dir_name = video_dir.name
        output_file = output_path / f"{dir_name}.json"

        # Skip if already processed
        if output_file.exists():
            print(f"[{i}/{len(all_dirs)}] ⏭️  {dir_name} - Already done, skip")
            continue

        print(f"[{i}/{len(all_dirs)}] 🔄 {dir_name}")

        traj_file = trajectory_path / f"{dir_name}.txt"

        try:
            description, success = process_one_video(client, video_dir, traj_file, MODEL)

            if success:
                result = {
                    "video_name": dir_name,
                    "action_label": get_action_label_from_dir(video_dir),
                    "description": description,
                    "model": MODEL
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                results.append(result)
                print(f"  ✓ Done: {description[:80]}...\n")
            else:
                print(f"  ✗ Failed\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")

    # Save all results
    if results:
        all_file = output_path / "all_descriptions.json"
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ All results saved to: {all_file}")

    print(f"\n✅ Complete! Processed {len(results)} videos")


if __name__ == "__main__":
    main()
