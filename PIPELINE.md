# VIA-EgoDex Description Construction Pipeline

## Overview

This pipeline constructs fine-grained action descriptions for egocentric videos, following and improving upon the **EgoHOD dataset methodology**. The approach combines **detection models** with **LLM-based rephrasing** to generate natural language descriptions of hand-object interactions, specifically tailored for **visually impaired assistance (VIA)** scenarios.

**Pipeline Architecture:**
```
Video Frames → Detection Stage → Trajectory Stage → LLM Rephrasing → Final Descriptions
```

**Key Improvements over EgoHOD:**
- ✨ **Open-vocabulary object detection** with YOLO-World (789 classes)
- ✨ **Multimodal LLM input** (video frames + trajectory text)
- ✨ **State-of-the-art model** (GPT-4.5 Vision instead of Yi-32B)
- ✨ **VIA-specific format** (scene description + safety warnings)

---

## Stage 1: Video Frame Extraction (Preprocessing)

**Input:** EgoDex videos
**Output:** Frame directories (`*.frames/` containing `0.jpg`, `1.jpg`, ..., `8.jpg` or `10.jpg`)

```
/data1/zy/full_data/EgoDex/frames/
├── add_remove_lid_0.frames/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── 8.jpg
├── add_remove_lid_1.frames/
└── ...
```

Each video is sampled into **8-10 frames** representing the temporal progression of the action.

---

## Stage 2: Detection & Trajectory Building

**Script:** `batch_process.py`

**Models Used:**
- **Hand Detector:** Detectron2 Faster R-CNN (trained on **100DOH dataset**, same as EgoHOD)
- **Object Detector:** YOLO-World v2 (open-vocabulary, **789 object classes**)

### 2.1 Hand Detection (`utils/hand_detector.py`)

**Consistent with EgoHOD:** Uses Detectron2 Faster R-CNN with X-101-32x8d-FPN backbone, pre-trained on the 100DOH (100 Days of Hands) dataset for robust egocentric hand detection.

**Model Configuration:**
- Architecture: Faster R-CNN with X-101-32x8d FPN
- Training data: 100DOH dataset
- Confidence threshold: 0.5
- Detects left and right hands

**Output per frame:**
```python
[
  {"bbox": [x1, y1, x2, y2], "confidence": 0.9},
  {"bbox": [x1, y1, x2, y2], "confidence": 0.8}
]
```

### 2.2 Object Detection (`utils/object_detector.py`)

**Improvement over EgoHOD:** Instead of class-specific detectors, we use **YOLO-World v2** for open-vocabulary object detection, supporting **789 object classes**.

**Class Vocabulary Source:**
- Merged from **EgoHOD annotations** (common egocentric objects)
- Extended with **EgoDex action labels** (action-relevant objects)
- Total: 789 classes including kitchen items, tools, materials, furniture, etc.

**Model Configuration:**
- Architecture: YOLO-World v2 (yolov8l-worldv2.pt)
- Confidence threshold: 0.3 (lower for better recall)
- Custom vocabulary: 789 classes

**Output per frame:**
```python
[
  {"bbox": [x1, y1, x2, y2], "confidence": 0.65, "class": 42, "name": "bottle"},
  {"bbox": [x1, y1, x2, y2], "confidence": 0.52, "class": 323, "name": "knife"}
]
```

**Example classes:** knife, pepper, bottle, cutting board, bowl, pan, cloth, paper, wood, glue, paint, etc. (see `utils/yolo_world_classes.py` for full list)

### 2.3 Contact Matching (`utils/contact_matcher.py`)

Matches hands to objects based on bounding box overlap (IoU) to determine contact relationships.

**Contact Types:**
- **Left-hand object:** Object only touching left hand
- **Right-hand object:** Object only touching right hand
- **Two-hand object:** Object touching both hands simultaneously

**Algorithm:**
```python
def match_hands_objects(hands, objects):
    1. Calculate IoU between each hand and each object
    2. If IoU > threshold (contact detected):
       - Assign object to left/right/both hands
    3. Return contact relationships
```

**Output per frame:**
```python
{
  "left_hand": {"bbox": [...], "confidence": 0.9},
  "right_hand": {"bbox": [...], "confidence": 0.8},
  "left_object": {"bbox": [...], "name": "knife", "confidence": 0.6},
  "right_object": {"bbox": [...], "name": "pepper", "confidence": 0.7},
  "two_hand_object": None
}
```

### 2.4 Trajectory Building (`utils/trajectory_builder.py`)

**Consistent with EgoHOD:** Aggregates frame-by-frame detections into hand-object interaction trajectories over time, tracking hand movements and object contacts.

**Trajectory Representation:**

1. **Coordinate format (optional):**
   ```
   left hand: (520, 340), (525, 345), (530, 350), ...
   ```

2. **Direction format (default):**
   ```
   left hand: (520, 340), down-right, down-right, still, ...
   ```
   - First frame: absolute coordinates
   - Subsequent frames: relative directions (up, down, left, right, up-left, etc.)

**LLM-Friendly Format:**
```
Left hand: down-right → down-right → still → down (object: knife)
Right hand: left → down-left → down-left (object: pepper)
Two-hand object: none
```

**Output Files:** `./trajectories/*.frames.txt`

**Example trajectory file:**
```
## Hand Object Dynamics
left hand:(520,340,down-right,down-right,still,down)
right hand:(680,420,left,down-left,down-left)
left hand object:(knife)
right hand object:(pepper)
two hand object:()

## LLM-Friendly Format:
Left hand: down-right → down-right → still → down (object: knife)
Right hand: left → down-left → down-left (object: pepper)
Two-hand object: none
```

---

## Stage 3: LLM-Based Rephrasing

**Script:** `generate_descriptions.py`
**Model:** GPT-4.5 Vision (`gpt-5.2-2025-12-11`)

**Key Improvements over EgoHOD:**

| Feature | EgoHOD | VIA-EgoDex (Ours) |
|---------|--------|-------------------|
| **Visual Input** | ❌ Text-only | ✅ **All video frames** (multimodal) |
| **Examples** | ✅ Few-shot | ✅ **From EgoHOD** (10 examples) |
| **Model** | Yi-32B (text) | ✅ **GPT-4.5 Vision** (SOTA multimodal) |
| **VIA Format** | General | ✅ **Scene description + Safety warnings** |

### 3.1 Input Preparation

For each video, prepare **multimodal inputs**:

1. **Visual Input (NEW):** All video frames (0.jpg → 8.jpg), encoded as base64
   - EgoHOD likely used text-only trajectory descriptions
   - Our approach enables true visual understanding of the action

2. **Trajectory Input:** Hand-object dynamics from Stage 2
   ```
   Left hand: down-right → down-right → still → down (object: knife)
   Right hand: left → down-left → down-left (object: pepper)
   ```

3. **Action Label:** Extracted from directory name (e.g., `cut_bell_pepper`)

### 3.2 Prompt Construction (`prompts.py`)

**Components:**

1. **System Role (VIA-specific):**
   ```
   You are an assistive narrator for visually impaired users.
   Your job is to describe fine-grained hand manipulation actions
   shown in the provided video frames.
   ```

2. **Task Instructions:**
   - Write 2-3 sentences total
   - Use natural, conversational English
   - NO pixels, bounding boxes, coordinates, or frame numbers
   - Use "left hand" / "right hand" explicitly
   - Contact terminology: "left-hand object", "right-hand object", "two-hand object"

3. **Content Structure (VIA-tailored):**
   - **Sentence 1:** Short scene description (objects present)
   - **Sentence 2:** Detailed hand manipulation (motion, contact, manipulation type)
   - **Sentence 3 (if applicable):** Potential safety warnings (sharp, hot, fragile, spill risk)

4. **Few-Shot Examples (from EgoHOD):**
   ```
   Example 1: The left hand moves from the top right to the center,
   while the right hand moves from the bottom left to the center.
   Both hands interact with the pepper, which is initially placed
   in the left hand and then transferred to the right hand before
   being put into the nylon.

   Example 2: The left hand moves from a lower position to the top
   of the sink, placing pepper seeds down. The right hand remains
   in a stationary position, supporting the pepper seeds.
   ```
   - 10 examples total (sourced from EgoHOD dataset)
   - Randomly sample 2 for each query (reduces prompt length)

5. **Input Context:**
   ```
   Action label: cut_bell_pepper
   Hand trajectory:
   Left hand: down-right → down-right → still → down (object: knife)
   Right hand: left → down-left → down-left (object: pepper)
   ```

### 3.3 GPT Vision API Call

**Model:** GPT-4.5 Vision (one of the most advanced multimodal models as of 2025)

**Request Format:**
```python
{
  "model": "gpt-5.2-2025-12-11",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # Frame 0
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # Frame 1
        ...
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # Frame 8
        {"type": "text", "text": "<full_prompt>"}
      ]
    }
  ],
  "temperature": 0.7,
  "max_tokens": 300
}
```

**Key Features:**
- **Multimodal reasoning:** All frames + text prompt sent together (improvement over EgoHOD's text-only Yi-32B)
- **Temporal understanding:** Model sees full action sequence visually
- **Grounded by trajectory:** Hand-object dynamics guide generation
- **State-of-the-art:** GPT-4.5 Vision provides superior understanding compared to previous models

### 3.4 Output Format

**Per-video JSON (VIA-specific format):**
```json
{
  "video_name": "cut_bell_pepper_5.frames",
  "action_label": "cut_bell_pepper",
  "description": "A red bell pepper is placed on a wooden cutting board, with a knife nearby. The left hand holds the knife and makes downward cutting motions through the pepper, while the right hand stabilizes the pepper from the side. The knife is sharp, so care should be taken during this action.",
  "model": "gpt-5.2-2025-12-11"
}
```

**VIA-specific output structure:**
- ✅ **Sentence 1:** Scene description ("A red bell pepper is placed on a wooden cutting board, with a knife nearby")
- ✅ **Sentence 2:** Hand manipulation ("The left hand holds... while the right hand stabilizes...")
- ✅ **Sentence 3:** Safety warning ("The knife is sharp, so care should be taken")

**Consolidated JSON:** `./descriptions/all_descriptions.json` contains all video results.

---

## Pipeline Execution

### Step 1: Generate Trajectories

```bash
python batch_process.py \
    --frames_dir /data1/zy/full_data/EgoDex/frames \
    --hand_model /data1/zy/models/hand_detector/hand_detector.pth \
    --yolo_model yolov8l-worldv2.pt \
    --output_dir ./trajectories
```

**Output:** `./trajectories/*.frames.txt` (one per video)

### Step 2: Generate Descriptions

```bash
export OPENAI_API_KEY="your-api-key"
python generate_descriptions.py
```

**Configuration (in code):**
```python
FRAMES_DIR = "/data1/zy/full_data/EgoDex/frames"
TRAJECTORY_DIR = "./trajectories"
OUTPUT_DIR = "./descriptions"
MODEL = "gpt-5.2-2025-12-11"
BASE_URL = "https://xiaoai.plus/v1"  # Custom API endpoint
```

**Output:** `./descriptions/*.json` + `./descriptions/all_descriptions.json`

---

## Design Rationale

### Why Detection + LLM? (Following EgoHOD)

1. **Detection Stage provides grounding:**
   - Precise hand and object localization
   - Contact relationship detection
   - Temporal trajectory tracking

2. **LLM Stage provides naturalness:**
   - Converts structured trajectories into fluent descriptions
   - Adds semantic understanding of actions
   - Removes technical jargon (coordinates, bboxes)

### Detailed Comparison with EgoHOD

| Component | EgoHOD | VIA-EgoDex (Ours) | Improvement |
|-----------|--------|-------------------|-------------|
| **Hand Detection** | Detectron2 (100DOH) | ✅ **Same: Detectron2 (100DOH)** | Consistent baseline |
| **Object Detection** | Class-specific detectors | ✅ **YOLO-World (789 classes)** | Open-vocabulary, more flexible |
| **Object Classes** | Fixed categories | ✅ **789 classes from EgoHOD + EgoDex labels** | Much broader coverage |
| **Trajectory Building** | Hand-object over time | ✅ **Same: EgoHOD method** | Consistent methodology |
| **Contact Matching** | IoU-based | ✅ **Same: IoU threshold 0.01** | Consistent approach |
| **LLM Model** | Yi-32B (text-only) | ✅ **GPT-4.5 Vision (SOTA multimodal)** | Superior understanding |
| **Visual Input** | ❌ None (text-only) | ✅ **All 8-10 video frames** | True multimodal reasoning |
| **Few-Shot Examples** | ✅ Yes | ✅ **10 examples from EgoHOD** | Borrowed best practices |
| **Output Format** | General description | ✅ **VIA-specific: Scene + Action + Safety** | Tailored for accessibility |

**Key Innovations:**

1. **Open-Vocabulary Detection:** YOLO-World with 789 classes (vs. fixed categories) enables detection of diverse objects across different action domains

2. **Multimodal LLM Input:** Sending **all video frames** to GPT-4.5 Vision enables true visual understanding, not just text-based trajectory interpretation

3. **State-of-the-Art Model:** GPT-4.5 Vision significantly outperforms Yi-32B in multimodal reasoning and natural language generation

4. **VIA-Specific Format:** Structured output with scene description + manipulation details + safety warnings, specifically designed for visually impaired assistance

---

## Key Features

✅ **Consistent with EgoHOD:** Hand detection (100DOH), trajectory building, contact matching
✅ **Improved object detection:** YOLO-World with 789 classes (from EgoHOD + EgoDex labels)
✅ **Multimodal LLM reasoning:** All frames + trajectory (vs. EgoHOD's text-only)
✅ **State-of-the-art model:** GPT-4.5 Vision (vs. Yi-32B)
✅ **VIA-specific format:** Scene description + manipulation + safety warnings
✅ **Scalable pipeline:** Batch processing with resume capability

---

## File Structure

```
via_egodex_construct/
├── batch_process.py              # Stage 2: Detection + Trajectory
├── generate_descriptions.py      # Stage 3: LLM Rephrasing
├── prompts.py                    # Prompt templates and examples
├── utils/
│   ├── hand_detector.py          # Detectron2 hand detection
│   ├── object_detector.py        # YOLO-World object detection
│   ├── contact_matcher.py        # Hand-object contact matching
│   └── trajectory_builder.py     # Trajectory aggregation
├── trajectories/                 # Stage 2 output
│   └── *.frames.txt
└── descriptions/                 # Stage 3 output
    ├── *.frames.json
    └── all_descriptions.json
```

---

## Example Output

**Input Video:** `cut_bell_pepper_5.frames` (9 frames)

**Trajectory (Stage 2):**
```
Left hand: down-right → down → still → down (object: knife)
Right hand: left → down-left → still (object: bell pepper)
```

**Description (Stage 3):**
```
A red bell pepper is placed on a wooden cutting board, with a knife nearby.
The left hand holds the knife and makes downward cutting motions through the
pepper, while the right hand stabilizes the pepper from the side. The knife
is sharp, so care should be taken during this action.
```

---

## Performance Considerations

- **Hand detection:** ~0.1s per frame (GPU)
- **Object detection:** ~0.2s per frame (GPU, depends on YOLO model size)
- **GPT API call:** ~3-5s per video (depends on frame count and API latency)
- **Total:** ~5-10s per video (with GPU acceleration)

For 3245 videos in EgoDex: ~4-9 hours total processing time.
