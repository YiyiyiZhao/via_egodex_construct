"""
Trajectory builder for hand and object dynamics
"""
from contact_matcher import get_bbox_center
import math


class TrajectoryBuilder:
    """Build trajectories from frame-by-frame detections"""

    def __init__(self, use_relative_direction=False):
        self.use_relative_direction = use_relative_direction

        self.left_hand_trajectory = []
        self.right_hand_trajectory = []
        self.left_object_trajectory = []
        self.right_object_trajectory = []
        self.two_hand_object_trajectory = []

        # Store object names
        self.left_object_names = []
        self.right_object_names = []
        self.two_hand_object_names = []

    def add_frame(self, match_result):
        """
        Add a frame's detection result to trajectories
        :param match_result: Output from match_hands_objects()
        """
        # Extract bounding boxes
        left_hand = match_result.get('left_hand')
        right_hand = match_result.get('right_hand')
        left_object = match_result.get('left_object')
        right_object = match_result.get('right_object')
        two_hand_object = match_result.get('two_hand_object')

        # Calculate center points and add to trajectories
        self.left_hand_trajectory.append(
            get_bbox_center(left_hand['bbox']) if left_hand else None
        )
        self.right_hand_trajectory.append(
            get_bbox_center(right_hand['bbox']) if right_hand else None
        )

        # For objects, store both center point and name
        self.left_object_trajectory.append(
            get_bbox_center(left_object['bbox']) if left_object else None
        )
        self.left_object_names.append(
            left_object.get('name', 'unknown') if left_object else None
        )

        self.right_object_trajectory.append(
            get_bbox_center(right_object['bbox']) if right_object else None
        )
        self.right_object_names.append(
            right_object.get('name', 'unknown') if right_object else None
        )

        self.two_hand_object_trajectory.append(
            get_bbox_center(two_hand_object['bbox']) if two_hand_object else None
        )
        self.two_hand_object_names.append(
            two_hand_object.get('name', 'unknown') if two_hand_object else None
        )

    def get_trajectories(self):
        """
        Get all trajectories
        :return: Dictionary of trajectories
        """
        return {
            'left_hand': self.left_hand_trajectory,
            'right_hand': self.right_hand_trajectory,
            'left_object': self.left_object_trajectory,
            'right_object': self.right_object_trajectory,
            'two_hand_object': self.two_hand_object_trajectory
        }

    def calculate_direction(self, p1, p2, threshold=10):
        """
        Calculate direction from p1 to p2
        :param p1: (x1, y1)
        :param p2: (x2, y2)
        :param threshold: Minimum distance to consider as movement
        :return: Direction string like "right", "up-left", "still"
        """
        if p1 is None or p2 is None:
            return "none"

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        distance = math.sqrt(dx**2 + dy**2)

        # Too small to consider movement
        if distance < threshold:
            return "still"

        # Calculate 8 directions
        angle = math.atan2(dy, dx)  # Note: y-axis is downward in images
        angle_deg = math.degrees(angle)

        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360

        # 8 directions (45 degrees each)
        if angle_deg < 22.5 or angle_deg >= 337.5:
            return "right"
        elif angle_deg < 67.5:
            return "down-right"
        elif angle_deg < 112.5:
            return "down"
        elif angle_deg < 157.5:
            return "down-left"
        elif angle_deg < 202.5:
            return "left"
        elif angle_deg < 247.5:
            return "up-left"
        elif angle_deg < 292.5:
            return "up"
        else:
            return "up-right"

    def trajectory_to_directions(self, trajectory):
        """
        Convert trajectory to direction sequence
        :param trajectory: List of (x, y) points
        :return: List of directions
        """
        if len(trajectory) < 2:
            return []

        directions = []
        for i in range(len(trajectory) - 1):
            direction = self.calculate_direction(trajectory[i], trajectory[i + 1])
            directions.append(direction)

        return directions

    def format_trajectory(self, trajectory, names=None):
        """
        Format trajectory as string
        If use_relative_direction=True: first frame as (x,y), then directions
        If use_relative_direction=False: all frames as (x,y)
        :param trajectory: List of (x, y) tuples or None
        :param names: Optional list of object names
        :return: Formatted string
        """
        if len(trajectory) == 0:
            return "()"

        if self.use_relative_direction:
            # First frame: absolute coordinates
            result = []

            # First point
            first_point = trajectory[0]
            if first_point is None:
                if names and names[0]:
                    result.append(f"(None,None,{names[0]})")
                else:
                    result.append("(None,None)")
            else:
                if names and names[0]:
                    result.append(f"({first_point[0]},{first_point[1]},{names[0]})")
                else:
                    result.append(f"({first_point[0]},{first_point[1]})")

            # Subsequent frames: directions
            directions = self.trajectory_to_directions(trajectory)
            for i, direction in enumerate(directions):
                if names and i + 1 < len(names) and names[i + 1]:
                    result.append(f"{direction},{names[i + 1]}")
                else:
                    result.append(direction)

            return "(" + ",".join(result) + ")"
        else:
            # Output all as absolute coordinates
            points = []
            for i, point in enumerate(trajectory):
                if point is None:
                    if names and i < len(names) and names[i]:
                        points.append(f"(None,None,{names[i]})")
                    else:
                        points.append("(None,None)")
                else:
                    if names and i < len(names) and names[i]:
                        points.append(f"({point[0]},{point[1]},{names[i]})")
                    else:
                        points.append(f"({point[0]},{point[1]})")
            return "(" + ",".join(points) + ")"

    def get_object_summary(self, object_names):
        """
        Summarize object changes across frames
        :param object_names: List of object names
        :return: Human-readable summary
        """
        if not object_names or all(name is None for name in object_names):
            return "none"

        # Remove None values
        valid_names = [name for name in object_names if name is not None]
        if not valid_names:
            return "none"

        # Check if object is constant
        unique_names = list(set(valid_names))
        if len(unique_names) == 1:
            return unique_names[0]

        # Multiple objects, show sequence
        return " → ".join(unique_names)

    def format_for_llm(self):
        """
        Format trajectories in a human-readable way for LLM prompts
        :return: Human-readable trajectory description
        """
        output = ""

        # Left hand
        if self.left_hand_trajectory:
            directions = self.trajectory_to_directions(self.left_hand_trajectory)
            direction_str = " → ".join(directions) if directions else "stationary"
            obj_summary = self.get_object_summary(self.left_object_names)
            output += f"Left hand: {direction_str} (object: {obj_summary})\n"

        # Right hand
        if self.right_hand_trajectory:
            directions = self.trajectory_to_directions(self.right_hand_trajectory)
            direction_str = " → ".join(directions) if directions else "stationary"
            obj_summary = self.get_object_summary(self.right_object_names)
            output += f"Right hand: {direction_str} (object: {obj_summary})\n"

        # Two-hand object
        two_hand_summary = self.get_object_summary(self.two_hand_object_names)
        if two_hand_summary != "none":
            output += f"Two-hand object: {two_hand_summary}\n"

        return output.strip()

    def format_all_trajectories(self):
        """
        Format all trajectories as the required output format
        :return: Formatted string
        """
        output = "## Hand Object Dynamics\n"
        output += f"left hand:{self.format_trajectory(self.left_hand_trajectory)}\n"
        output += f"right hand:{self.format_trajectory(self.right_hand_trajectory)}\n"
        output += f"left hand object:{self.format_trajectory(self.left_object_trajectory, self.left_object_names)}\n"
        output += f"right hand object:{self.format_trajectory(self.right_object_trajectory, self.right_object_names)}\n"
        output += f"two hand object:{self.format_trajectory(self.two_hand_object_trajectory, self.two_hand_object_names)}\n"
        return output
