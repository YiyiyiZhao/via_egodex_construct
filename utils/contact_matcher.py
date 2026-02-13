def compute_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def get_bbox_center(bbox):
    """
    Get center point of a bounding box
    :param bbox: [x1, y1, x2, y2]
    :return: (center_x, center_y)
    """
    if bbox is None:
        return None
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return (int(round(center_x)), int(round(center_y)))


def distinguish_hands(hands):
    """
    Distinguish left and right hands based on position
    Left side of image = left hand, right side = right hand
    """
    if len(hands) == 0:
        return None, None
    if len(hands) == 1:
        hand = hands[0]
        return hand, None
    hands_sorted = sorted(hands, key=lambda h: (h['bbox'][0] + h['bbox'][2]) / 2)
    left_hand = hands_sorted[0]
    right_hand = hands_sorted[-1]
    return left_hand, right_hand


def match_hand_to_object(hand, objects, iou_threshold=0.01):
    """
    Find the contacted object for a hand (the one with maximum IoU)
    :param hand: {'bbox': [x1,y1,x2,y2], ...}
    :param objects: [{'bbox': [x1,y1,x2,y2], 'name': 'cup', ...}, ...]
    :param iou_threshold: IoU threshold, contact is determined if exceeded
    :return: Contacted object dict, or None
    """
    if hand is None or len(objects) == 0:
        return None

    max_iou = 0
    matched_obj = None

    for obj in objects:
        iou = compute_iou(hand['bbox'], obj['bbox'])
        if iou > max_iou and iou > iou_threshold:
            max_iou = iou
            matched_obj = obj

    return matched_obj


def find_two_hand_object(left_hand, right_hand, objects, iou_threshold=0.01):
    """
    Find object that is contacted by both hands
    :param left_hand: Left hand detection result
    :param right_hand: Right hand detection result
    :param objects: List of detected objects
    :param iou_threshold: IoU threshold
    :return: Object contacted by both hands, or None
    """
    if left_hand is None or right_hand is None or len(objects) == 0:
        return None

    for obj in objects:
        left_iou = compute_iou(left_hand['bbox'], obj['bbox'])
        right_iou = compute_iou(right_hand['bbox'], obj['bbox'])

        # Both hands have contact with this object
        if left_iou > iou_threshold and right_iou > iou_threshold:
            return obj

    return None


def match_hands_objects(hands, objects, iou_threshold=0.01):
    """
    Complete matching: distinguish left/right hands + find contacted objects for each

    :param hands: List of hand detection results
    :param objects: List of object detection results
    :param iou_threshold: IoU threshold
    :return: {
        'left_hand': {...} or None,
        'right_hand': {...} or None,
        'left_object': {...} or None,
        'right_object': {...} or None,
        'two_hand_object': {...} or None
    }
    """
    # Distinguish left and right hands
    left_hand, right_hand = distinguish_hands(hands)

    # Match objects to each hand
    left_object = match_hand_to_object(left_hand, objects, iou_threshold)
    right_object = match_hand_to_object(right_hand, objects, iou_threshold)

    # Find object contacted by both hands
    two_hand_object = find_two_hand_object(left_hand, right_hand, objects, iou_threshold)

    return {
        'left_hand': left_hand,
        'right_hand': right_hand,
        'left_object': left_object,
        'right_object': right_object,
        'two_hand_object': two_hand_object
    }
