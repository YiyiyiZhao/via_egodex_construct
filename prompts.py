import  random
import os
import re

def get_action_label_from_dir(dir_path):
    name = os.path.basename(os.path.normpath(dir_path))
    name = re.sub(r'\.frames$', '', name)
    name = re.sub(r'_\d+$', '', name)
    action_label = name
    return action_label

def construct_examples():
    example_1="""The left hand moves from the top right to the center, while the right hand moves from the bottom left to the center. Both hands interact with the pepper, which is initially placed in the left hand and then transferred to the right hand before being put into the nylon."""
    example_2="""The left hand moves from a lower position to the top of the sink, placing pepper seeds down. The right hand remains in a stationary position, supporting the pepper seeds."""
    example_3="""The left hand moves downward and slightly to the left, while the right hand moves upward and slightly to the right. The object held by the left hand remains stationary, while the object held by the right hand moves in an arc from above the piece of wood to a position where it is being used to mark the wood."""
    example_4="""The left hand rotates the dough clockwise on the tray while the right hand gently sifts flour over it."""
    example_5="""The left hand moves from the left to the center, while the right hand brings an eggplant and places it on the chop board."""
    example_6="""The left hand moves towards the knife, while the right hand approaches the object from the side. The knife is grasped by the left hand, and the object is held by the right hand."""
    example_7="""The left hand moves from bottom left to top right, gathering vegetables. The right hand supports the object from the top, with both hands working together to pack the vegetables."""
    example_8="""The left hand moves in a downward motion, while the right hand maintains a stable position. The object in the left hand is being wiped with the knife, which remains stationary in the right hand."""
    example_9="""The left hand moves downwards to press buttons on the drum pad machine, while the right hand adjusts the settings."""
    example_10="""The left hand grasps the paintbrush and moves it downwards to dip into the paint can, while the right hand steadies the can."""
    example_list = [example_1, example_2, example_3, example_4,example_5,example_6,example_7,example_8,example_9,example_10]
    random.shuffle(example_list)
    numbered_examples = [f"{i}. {ex}" for i, ex in enumerate(example_list, 1)]
    examples = "Examples:\n" + "\n\n".join(numbered_examples[:2])
    return examples


def construct_task_prompt(examples,action_label,hand_trajectory):
    prompt = f"""You are an assistive narrator for visually impaired users. Your job is to describe the fine-grained hand manipulation action shown in the provided video frames.

    You will be given:
    1) Frames over time (in chronological order).
    2) Action label: a very concise description of what is happening. You MUST follow it closely.
    3) Hand trajectory over frames (for reference).

    Output requirements:
    - Write 2–3 sentences total. No headings, no bullet points, no extra words.
    - Use natural, conversational English.
    - Do NOT mention pixels, bounding boxes, coordinates, or frame numbers.
    - Use "left hand" / "right hand" explicitly when needed.
    - Contact terminology:
      * "two-hand object": the object is in contact with BOTH hands
      * "left-hand object": in contact with the left hand
      * "right-hand object": in contact with the right hand

    Content structure (must follow):
    Sentence 1: Briefly describe what objects are in front.
    Sentence 2: Describe the hand manipulation in detail, including:
      - direction of motion (for each hand if visible)
      - relative relationship to objects (approaching, touching, holding, releasing, rotating, sliding, etc.)
      - whether each hand is touching or not, and which object is the left-hand/right-hand/two-hand object (if applicable)
    Sentence 3 (optional, only if applicable): Note safety concerns (e.g., hot, sharp, fragile, spill risk). If no safety concern, omit this sentence.

    Examples of Sentence 2 style:{examples}
    Action label: {action_label}
    Hand trajectory: {hand_trajectory}
    
    Now generate the 2–3 sentence description:
    """
    return prompt








