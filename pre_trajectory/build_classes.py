"""
Build YOLO-World object classes from EgoDex dataset
Strategy: Sample captions from CSV + Extract actions from directory names + GPT extraction
"""
import csv
import random
import os
import re
import json
from pathlib import Path
from typing import Set, List
import openai
import pdb

def sample_captions_from_csv(csv_file: str, n: int = 2000) -> List[str]:
    """
    Randomly sample n captions from the huge CSV file
    Uses reservoir sampling to handle large files efficiently
    """
    print(f"Sampling {n} captions from {csv_file}...")

    sampled_captions = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Reservoir sampling
        for i, row in enumerate(reader):
            caption = row['caption']

            if i < n:
                sampled_captions.append(caption)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < n:
                    sampled_captions[j] = caption

            # Progress indicator
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1} rows...")

    print(f"✓ Sampled {len(sampled_captions)} captions")
    return sampled_captions


def extract_actions_from_directories(frames_dir: str) -> List[str]:
    """
    Extract action names from directory names
    Example: "add_remove_lid_0.frames" -> "add_remove_lid"
    """
    print(f"Extracting actions from {frames_dir}...")

    frames_path = Path(frames_dir)
    if not frames_path.exists():
        print(f"Warning: {frames_dir} does not exist, skipping directory extraction")
        return []

    action_names = set()

    for dir_name in os.listdir(frames_dir):
        if dir_name.endswith('.frames'):
            # Remove .frames extension
            name = dir_name.replace('.frames', '')
            # Remove trailing numbers (e.g., _0, _12)
            name = re.sub(r'_\d+$', '', name)
            action_names.add(name)

    action_list = sorted(action_names)
    print(f"✓ Found {len(action_list)} unique actions")
    return action_list


def extract_objects_from_texts_with_gpt(texts: List[str], batch_size: int = 20,
                                        text_type: str = "caption") -> Set[str]:
    """
    Use GPT API to extract physical objects from text descriptions
    Process in batches to reduce API calls
    """
    print(f"\nExtracting objects from {len(texts)} {text_type}s using GPT...")
    print(f"Batch size: {batch_size}")

    all_objects = set()

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"  Processing batch {batch_num}/{total_batches}...")

        # Construct prompt
        if text_type == "caption":
            prompt = f"""Extract all physical objects mentioned in these action captions.
Return a JSON array of object names in lowercase, singular form.
Only include tangible objects (tools, containers, food items, furniture, etc.).
Exclude: hands, person, body parts, abstract concepts.

Captions:
{json.dumps(batch, indent=2)}

Return format: {{"objects": ["object1", "object2", ...]}}
"""
        else:  # action names
            prompt = f"""Given these action names, infer all possible physical objects that might be involved.
Return a JSON array of object names in lowercase, singular form.

Examples:
- "cut_vegetable" -> ["knife", "cutting board", "carrot", "onion", "tomato", "pepper"]
- "pour_water" -> ["bottle", "cup", "glass", "pitcher", "water", "jug"]
- "add_remove_lid" -> ["lid", "pot", "jar", "container", "pan"]

Action names:
{json.dumps(batch, indent=2)}

Return format: {{"objects": ["object1", "object2", ...]}}
"""

        try:
            # Call GPT API
            response = openai.ChatCompletion.create(
                model="gpt-5.2-2025-12-11",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts object names from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Extract JSON
            try:
                result = json.loads(content)
                objects = result.get('objects', [])
                all_objects.update(objects)
                print(f"    Found {len(objects)} objects in this batch")
            except json.JSONDecodeError:
                # Fallback: try to extract objects from plain text
                print(f"    Warning: JSON parsing failed, trying plain text extraction")
                objects = [obj.strip().lower() for obj in content.split('\n') if obj.strip()]
                all_objects.update(objects)

        except Exception as e:
            print(f"    Error processing batch {batch_num}: {e}")
            continue

    print(f"✓ Extracted {len(all_objects)} unique objects")
    return all_objects


def clean_and_filter_objects(objects: Set[str]) -> Set[str]:
    """
    Clean and filter the object list
    - Remove duplicates
    - Remove invalid entries
    - Normalize names
    """
    print("\nCleaning object list...")

    cleaned = set()

    # Blacklist: things we don't want
    blacklist = {
        'hand', 'hands', 'person', 'people', 'left hand', 'right hand',
        'finger', 'fingers', 'arm', 'arms', 'body',
        'c', '#c', 'object', 'thing', 'item'
    }

    for obj in objects:
        obj = obj.strip().lower()

        # Skip empty, too short, or blacklisted
        if len(obj) < 2 or obj in blacklist:
            continue

        # Skip if contains special characters (likely parsing errors)
        if not re.match(r'^[a-z0-9\s-]+$', obj):
            continue

        cleaned.add(obj)

    print(f"✓ Cleaned: {len(objects)} -> {len(cleaned)} objects")
    return cleaned


def save_classes(objects: Set[str], output_file: str):
    """
    Save object classes to a file
    """
    print(f"\nSaving classes to {output_file}...")

    sorted_objects = sorted(objects)

    with open(output_file, 'w') as f:
        for obj in sorted_objects:
            f.write(f"{obj}\n")

    print(f"✓ Saved {len(sorted_objects)} classes")

    # Also save as Python list for easy import
    py_file = output_file.replace('.txt', '.py')
    with open(py_file, 'w') as f:
        f.write("# Auto-generated object classes for YOLO-World\n")
        f.write("YOLO_WORLD_CLASSES = [\n")
        for obj in sorted_objects:
            f.write(f"    \"{obj}\",\n")
        f.write("]\n")

    print(f"✓ Also saved as Python list: {py_file}")


def main():
    """
    Main workflow
    """
    print("=" * 60)
    print("Building YOLO-World Classes for EgoDex Dataset")
    print("=" * 60)

    # Configuration
    CSV_FILE = "ego_hod.csv"
    FRAMES_DIR = "/data1/zy/full_data/EgoDex/frames"
    OUTPUT_FILE = "yolo_world_classes.txt"

    SAMPLE_SIZE = 2000  # Number of captions to sample
    BATCH_SIZE = 20     # GPT API batch size

    # OpenAI API Configuration
    # openai.api_key = "sk-xxxxxxxxx"
    # openai.api_base = "https://xxxxxxxxxxx"

    all_objects = set()

    # Step 1: Sample captions from CSV
    print("\n" + "=" * 60)
    print("STEP 1: Sample Captions from CSV")
    print("=" * 60)
    captions = sample_captions_from_csv(CSV_FILE, n=SAMPLE_SIZE)


    # Step 2: Extract actions from directory names
    print("\n" + "=" * 60)
    print("STEP 2: Extract Actions from Directory Names")
    print("=" * 60)
    actions = extract_actions_from_directories(FRAMES_DIR)

    # Step 3: Extract objects from captions using GPT
    print("\n" + "=" * 60)
    print("STEP 3: Extract Objects from Captions (GPT)")
    print("=" * 60)
    objects_from_captions = extract_objects_from_texts_with_gpt(captions, batch_size=BATCH_SIZE, text_type="caption")
    all_objects.update(objects_from_captions)


    # Step 4: Extract objects from actions using GPT
    print("\n" + "=" * 60)
    print("STEP 4: Extract Objects from Actions (GPT)")
    print("=" * 60)
    objects_from_actions = extract_objects_from_texts_with_gpt(
        actions, batch_size=BATCH_SIZE, text_type="action"
    )
    all_objects.update(objects_from_actions)

    # Step 5: Clean and filter
    print("\n" + "=" * 60)
    print("STEP 5: Clean and Filter Objects")
    print("=" * 60)
    cleaned_objects = clean_and_filter_objects(all_objects)

    # Step 6: Save to file
    print("\n" + "=" * 60)
    print("STEP 6: Save Classes")
    print("=" * 60)
    save_classes(cleaned_objects, OUTPUT_FILE)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Captions sampled: {len(captions)}")
    print(f"Actions extracted: {len(actions)}")
    print(f"Total unique objects: {len(cleaned_objects)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nFirst 20 classes:")
    for obj in sorted(cleaned_objects)[:20]:
        print(f"  - {obj}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
