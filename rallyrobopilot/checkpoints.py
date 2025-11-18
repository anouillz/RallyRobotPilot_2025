from ursina import Entity, Text, color, destroy, invoke, lerp
import json
import time
from pathlib import Path

MAX_RAYCAST_DIST = 100  # Same as in raycast_sensor.py


class CheckpointHandler:
    def __init__(
        self, track_name="SimpleTrack", track_scale=(1, 1, 1), checkpoint_size=25
    ):
        self.track_name = track_name
        self.track_scale = track_scale
        self.checkpoint_size = checkpoint_size  # Width of checkpoint crossing area
        self.checkpoints = []
        self.checkpoint_entities = []
        self.checkpoint_counter = 0
        self.passed_checkpoints = set()  # Track which checkpoints have been passed

        # Lap tracking
        self.current_lap = 0
        self.lap_checkpoints = []  # Ordered list of checkpoint IDs for lap completion
        self.next_checkpoint_index = 0  # Index of next checkpoint to pass
        self.lap_completed_checkpoints = (
            False  # Whether all checkpoints have been passed
        )

        # Track previous car position for crossing detection
        self.previous_car_position = None

        # UI elements
        self.feedback_text = Text(
            text="", position=(0, 0.35), scale=0.8, color=color.green
        )
        self.feedback_text.disable()  # Start disabled

        self.lap_text = Text(
            text="",
            position=(0, 0.45),
            scale=(0.8, 0.8),
            color=color.white,
            origin=(0.0, 0.0),
            size=0.05,
        )
        self.lap_text.disable()  # Start disabled

        self.ui_enabled = False  # Flag to control UI visibility

        # Load existing checkpoints if any
        self.load_checkpoints()

    def _complete_lap(self):
        """Handle lap completion"""
        lap_number = self.current_lap + 1
        self.current_lap += 1
        self.next_checkpoint_index = 0
        self.lap_completed_checkpoints = False

        # Reset all checkpoints for next lap
        for entity_data in self.checkpoint_entities:
            entity_data["passed"] = False
            entity_data["entity"].color = color.green
            entity_data["text"].color = color.yellow

        self.passed_checkpoints.clear()

        # Mark the first checkpoint as passed for the new lap
        if self.lap_checkpoints:
            first_id = self.lap_checkpoints[0]
            for entity_data in self.checkpoint_entities:
                if entity_data["data"]["id"] == first_id:
                    entity_data["passed"] = True
                    entity_data["entity"].color = color.blue
                    entity_data["text"].color = color.cyan
                    break
            self.passed_checkpoints.add(first_id)
            self.next_checkpoint_index = 1

        print(
            f"Lap {lap_number} completed! Returned to start after passing all checkpoints."
        )

    def get_lap_info(self):
        """Get current lap information"""
        return {
            "current_lap": self.current_lap,
            "next_checkpoint": self.lap_checkpoints[self.next_checkpoint_index]
            if self.lap_checkpoints
            and self.next_checkpoint_index < len(self.lap_checkpoints)
            else None,
            "checkpoints_passed": len(self.passed_checkpoints),
            "total_checkpoints": len(self.lap_checkpoints),
        }

    def create_checkpoint_entity(self, checkpoint_data):
        """Create a visual entity for the checkpoint"""
        pos = tuple(checkpoint_data["position"])
        checkpoint_size = checkpoint_data.get("size", self.checkpoint_size)

        # Ensure the checkpoint is perfectly centered
        center_position = (pos[0], pos[1], pos[2])

        # Create a simple visual line across the track (no collider)
        # Make it more visible with better proportions
        entity = Entity(
            model="cube",
            color=color.green,
            position=center_position,
            rotation=checkpoint_data["rotation"],
            scale=(checkpoint_size, 1, 0.2),  # Width, height, thickness
        )

        # Add a text label showing checkpoint number - position it above the checkpoint center
        text_entity = Text(
            text=f"CP{checkpoint_data['id']}",
            position=(center_position[0], center_position[1] + 1.5, center_position[2]),
            scale=1.2,
            color=color.yellow,
            billboard=True,
        )

        self.checkpoint_entities.append(
            {
                "entity": entity,
                "text": text_entity,
                "data": checkpoint_data,
                "passed": False,
                "creation_time": time.time(),
            }
        )

        # Animate the checkpoint appearance with a nice scale effect
        entity.scale = (0, 0, 0)
        invoke(entity.animate_scale, (checkpoint_size, 1, 0.2), duration=0.5)

    def remove_last_checkpoint(self):
        """Remove the last placed checkpoint"""
        if self.checkpoints:
            last_checkpoint = self.checkpoints.pop()
            if (
                self.lap_checkpoints
                and last_checkpoint["id"] == self.lap_checkpoints[-1]
            ):
                self.lap_checkpoints.pop()  # Remove from lap sequence

            if self.checkpoint_entities:
                last_entity_data = self.checkpoint_entities.pop()
                destroy(last_entity_data["entity"])
                destroy(last_entity_data["text"])

            self.checkpoint_counter -= 1

            # Reset lap progress if needed
            if self.next_checkpoint_index >= len(self.lap_checkpoints):
                self.next_checkpoint_index = 0

            self.save_checkpoints()
            self.update_status_text()
            print(f"Removed checkpoint {last_checkpoint['id']}")

    def clear_all_checkpoints(self):
        """Clear all checkpoints"""
        for entity_data in self.checkpoint_entities:
            destroy(entity_data["entity"])
            destroy(entity_data["text"])

        self.checkpoints.clear()
        self.checkpoint_entities.clear()
        self.lap_checkpoints.clear()
        self.passed_checkpoints.clear()
        self.checkpoint_counter = 0
        self.next_checkpoint_index = 0
        self.current_lap = 0
        self.lap_completed_checkpoints = False
        self.save_checkpoints()
        self.update_status_text()
        print("All checkpoints cleared")

    def get_checkpoint_positions(self):
        """Get list of checkpoint positions"""
        return [cp["position"] for cp in self.checkpoints]

    def save_checkpoints(self):
        """Save checkpoints to file"""
        root_dir = Path(__file__).resolve().parent.parent
        checkpoint_file = root_dir / f"assets/{self.track_name}/checkpoints.json"

        # Ensure directory exists
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_file, "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def load_checkpoints(self):
        """Load checkpoints from file"""
        root_dir = Path(__file__).resolve().parent.parent
        checkpoint_file = root_dir / f"assets/{self.track_name}/checkpoints.json"

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    self.checkpoints = json.load(f)

                # Update counter to next available ID
                if self.checkpoints:
                    self.checkpoint_counter = (
                        max(cp["id"] for cp in self.checkpoints) + 1
                    )

                # Create entities for loaded checkpoints
                for checkpoint_data in self.checkpoints:
                    self.create_checkpoint_entity(checkpoint_data)
                    # Mark loaded checkpoints as passable immediately
                    if self.checkpoint_entities:
                        self.checkpoint_entities[-1]["creation_time"] = 0

                # Populate lap checkpoints from loaded data
                self.lap_checkpoints = [cp["id"] for cp in self.checkpoints]

                print(f"Loaded {len(self.checkpoints)} checkpoints")
            except Exception as e:
                print(f"Error loading checkpoints: {e}")
                self.checkpoints = []
        else:
            print("No checkpoint file found, starting fresh")

    def place_checkpoint(self, position, rotation=(0, 0, 0), raycast_sensor=None):
        """Place a checkpoint at the given position"""

        # Calculate centered position and size based on raycasts
        centered_position = list(position)
        checkpoint_size = self.checkpoint_size

        if raycast_sensor:
            centered_position, checkpoint_size = (
                self._calculate_centered_position_and_size(
                    position, rotation, raycast_sensor
                )
            )

        checkpoint_data = {
            "id": self.checkpoint_counter,
            "position": centered_position,
            "rotation": list(rotation),
            "size": checkpoint_size,
        }

        self.checkpoints.append(checkpoint_data)
        self.lap_checkpoints.append(checkpoint_data["id"])  # Add to lap sequence
        self.checkpoint_counter += 1

        # Create visual entity
        self.create_checkpoint_entity(checkpoint_data)

        # Save checkpoints immediately
        self.save_checkpoints()

        self.update_status_text()

        print(
            f"Checkpoint {checkpoint_data['id']} placed at {centered_position} with size {checkpoint_size}"
        )

    def _calculate_centered_position_and_size(self, position, rotation, raycast_sensor):
        """Calculate centered position and size based on raycast measurements"""
        try:
            if hasattr(raycast_sensor, "rays") and raycast_sensor.rays:
                # Get the closest hits on left and right sides
                left_distances = []
                right_distances = []
                left_angles = []
                right_angles = []

                num_rays = len(raycast_sensor.rays)
                half_angle = getattr(raycast_sensor, "half_angle", 45)

                for i, ray in enumerate(raycast_sensor.rays):
                    if hasattr(ray, "sensing_dist"):
                        dist = ray.sensing_dist
                        if dist < MAX_RAYCAST_DIST:  # Only consider actual hits
                            # Calculate ray angle
                            angle = -half_angle + 2 * half_angle / (num_rays - 1) * i
                            if angle < -5:  # Left side (with small deadzone)
                                left_distances.append(dist)
                                left_angles.append(angle)
                            elif angle > 5:  # Right side (with small deadzone)
                                right_distances.append(dist)
                                right_angles.append(angle)

                if left_distances and right_distances:
                    # Find the closest hits on each side
                    left_min_idx = left_distances.index(min(left_distances))
                    right_min_idx = right_distances.index(min(right_distances))

                    left_min_dist = left_distances[left_min_idx]
                    right_min_dist = right_distances[right_min_idx]
                    left_angle = left_angles[left_min_idx]
                    right_angle = right_angles[right_min_idx]

                    # Calculate track center offset from car position
                    # The car is at position, track edges are at left_min_dist and right_min_dist
                    # along the directions given by the angles

                    import math

                    rot_y = rotation[1] if len(rotation) > 1 else 0

                    # Convert angles to world directions relative to car facing
                    left_world_angle = math.radians(rot_y + left_angle)
                    right_world_angle = math.radians(rot_y + right_angle)

                    # Calculate offset to track center
                    # Track center is midway between left and right edges
                    left_offset_x = left_min_dist * math.sin(left_world_angle)
                    left_offset_z = left_min_dist * math.cos(left_world_angle)
                    right_offset_x = right_min_dist * math.sin(right_world_angle)
                    right_offset_z = right_min_dist * math.cos(right_world_angle)

                    # Center is average of left and right edge positions
                    center_offset_x = (left_offset_x + right_offset_x) / 2
                    center_offset_z = (left_offset_z + right_offset_z) / 2

                    # Calculate centered position
                    centered_position = [
                        position[0] + center_offset_x,
                        position[1],
                        position[2] + center_offset_z,
                    ]

                    # Calculate track width
                    track_width = left_min_dist + right_min_dist
                    final_size = max(min(track_width, 50), 15)

                    print(
                        f"Track center offset: ({center_offset_x:.2f}, {center_offset_z:.2f})"
                    )
                    print(f"Centered position: {centered_position}")
                    print(f"Track width: {track_width}, Final size: {final_size}")

                    return centered_position, final_size

        except Exception as e:
            print(f"Error calculating centered position: {e}")

        # Fallback to original position and default size
        return list(position), self.checkpoint_size


    def flash_feedback(self, message):
        """Show temporary feedback message"""
        if hasattr(self, "feedback_text"):
            self.feedback_text.text = message
            self.feedback_text.enable()
            invoke(self.feedback_text.disable, delay=2.0)

    def update_status_text(self):
        """Update the status text with current lap info"""
        # Removed status text
        pass

    def show_ui(self):
        """Show checkpoint UI"""
        if self.ui_enabled:
            self.update_status_text()
            for entity_data in self.checkpoint_entities:
                entity_data["entity"].enable()
                entity_data["text"].enable()
            self.lap_text.enable()

    def hide_ui(self):
        """Hide checkpoint UI"""
        self.feedback_text.disable()
        self.lap_text.disable()
        for entity_data in self.checkpoint_entities:
            entity_data["entity"].disable()
            entity_data["text"].disable()

    def check_passed_checkpoints(self, car_position):
        """Check if car has passed through any checkpoints"""
        current_time = time.time()

        # Need previous position for crossing detection
        if self.previous_car_position is None:
            self.previous_car_position = car_position
            return

        for entity_data in self.checkpoint_entities:
            checkpoint_pos = entity_data["data"]["position"]
            checkpoint_rot = entity_data["data"]["rotation"]
            checkpoint_size = entity_data["data"].get("size", self.checkpoint_size)

            # Check if car crossed the checkpoint line
            if self._did_cross_checkpoint(
                self.previous_car_position,
                car_position,
                checkpoint_pos,
                checkpoint_rot,
                checkpoint_size,
            ):
                checkpoint_id = entity_data["data"]["id"]

                # Only allow passing after 1 second from creation (to prevent immediate passing)
                if current_time - entity_data["creation_time"] < 1.0:
                    continue

                # Check if this completes a lap (returning to checkpoint 0 after all others passed)
                if (
                    self.lap_completed_checkpoints
                    and checkpoint_id == self.lap_checkpoints[0]
                ):
                    entity_data["passed"] = True
                    entity_data[
                        "entity"
                    ].color = color.blue  # Change to blue when passed
                    entity_data["text"].color = color.cyan
                    self.passed_checkpoints.add(checkpoint_id)
                    self._complete_lap()
                    print(f"Passed checkpoint {checkpoint_id}")
                # Check if this is the next expected checkpoint in sequence
                elif (
                    not entity_data["passed"]
                    and self.lap_checkpoints
                    and checkpoint_id
                    == self.lap_checkpoints[self.next_checkpoint_index]
                ):
                    entity_data["passed"] = True
                    entity_data[
                        "entity"
                    ].color = color.blue  # Change to blue when passed
                    entity_data["text"].color = color.cyan
                    self.passed_checkpoints.add(checkpoint_id)

                    # Advance to next checkpoint
                    self.next_checkpoint_index += 1

                    print(f"Passed checkpoint {checkpoint_id}")

                    # Check if all checkpoints have been passed (but lap not yet completed)
                    if self.next_checkpoint_index >= len(self.lap_checkpoints):
                        self.lap_completed_checkpoints = True
                        print(
                            "All checkpoints passed - return to start to complete lap"
                        )
                else:
                    # Wrong checkpoint order - maybe show different feedback
                    if self.lap_completed_checkpoints:
                        expected = (
                            f"checkpoint {self.lap_checkpoints[0]} to complete lap"
                        )
                    elif not entity_data["passed"]:
                        expected = (
                            self.lap_checkpoints[self.next_checkpoint_index]
                            if self.lap_checkpoints
                            and self.next_checkpoint_index < len(self.lap_checkpoints)
                            else "none"
                        )
                    else:
                        expected = "checkpoint not passed yet"
                    print(
                        f"Wrong checkpoint order: got {checkpoint_id}, expected {expected}"
                    )

        self.previous_car_position = car_position

    def _did_cross_checkpoint(
        self, prev_pos, curr_pos, checkpoint_pos, checkpoint_rot, checkpoint_size
    ):
        """Check if car crossed the checkpoint line between prev_pos and curr_pos"""
        import math

        # Convert positions to tuples
        prev = tuple(prev_pos) if hasattr(prev_pos, "__iter__") else prev_pos
        curr = tuple(curr_pos) if hasattr(curr_pos, "__iter__") else curr_pos
        cp = tuple(checkpoint_pos)

        # Get checkpoint orientation (assuming rotation is around Y axis)
        rot_y = checkpoint_rot[1] if len(checkpoint_rot) > 1 else 0

        # Calculate perpendicular vector to checkpoint line (line direction)
        perp_x = math.cos(math.radians(rot_y))
        perp_z = math.sin(math.radians(rot_y))

        # Forward vector is the crossing direction (perpendicular to line)
        forward_x = math.sin(math.radians(rot_y))
        forward_z = math.cos(math.radians(rot_y))

        # Vector from checkpoint center to previous position
        prev_vec = (prev[0] - cp[0], 0, prev[2] - cp[2])
        # Vector from checkpoint center to current position
        curr_vec = (curr[0] - cp[0], 0, curr[2] - cp[2])

        # Movement vector
        move_vec = (curr[0] - prev[0], 0, curr[2] - prev[2])

        # Dot products with forward vector (for side detection, distance to the plane)
        prev_side = prev_vec[0] * forward_x + prev_vec[2] * forward_z
        curr_side = curr_vec[0] * forward_x + curr_vec[2] * forward_z

        # Check if movement is generally towards the checkpoint
        move_dot_forward = move_vec[0] * forward_x + move_vec[2] * forward_z

        # Check if signs are different (crossed the line) and within checkpoint size
        checkpoint_width = checkpoint_size / 2  # Half width along the line length
        checkpoint_thickness = 0.2  # Thickness of the visual line
        prev_distance = abs(prev_side)
        curr_distance = abs(curr_side)

        # Calculate the exact crossing point
        if prev_side * curr_side >= 0:
            return False  # No crossing occurred

        # Interpolate to find crossing point
        # prev_side + t * (curr_side - prev_side) = 0
        # t = -prev_side / (curr_side - prev_side)
        t = -prev_side / (curr_side - prev_side) if curr_side != prev_side else 0

        # Clamp t to [0, 1] to ensure it's between prev and curr positions
        t = max(0, min(1, t))

        # Calculate crossing point
        cross_x = prev[0] + t * (curr[0] - prev[0])
        cross_z = prev[2] + t * (curr[2] - prev[2])

        # Check if crossing point is within checkpoint bounds
        # The checkpoint has:
        # - Width: checkpoint_size (along the line length)
        # - Thickness: 0.2 (along the crossing direction)

        # Vector from checkpoint center to crossing point
        cross_vec = (cross_x - cp[0], 0, cross_z - cp[2])

        # Distance along line length axis (width check)
        cross_distance_width = abs(cross_vec[0] * perp_x + cross_vec[2] * perp_z)

        # Distance along crossing axis (thickness check)
        cross_distance_length = abs(cross_vec[0] * forward_x + cross_vec[2] * forward_z)

        # Check if crossing point is within both width and length bounds
        within_width = cross_distance_width <= checkpoint_width
        within_length = cross_distance_length <= checkpoint_thickness

        # Check movement direction
        moving_towards = move_dot_forward > 0

        return within_width and within_length and moving_towards

    def update(self):
        """Update method to be called in the game loop"""
        if self.ui_enabled:
            lap_info = self.get_lap_info()
            if lap_info["total_checkpoints"] > 0:
                self.lap_text.text = f"Lap {lap_info['current_lap'] + 1}: {lap_info['checkpoints_passed']}/{lap_info['total_checkpoints']}"
            else:
                self.lap_text.disable()
        else:
            self.lap_text.disable()
        # Removed glow animation for simplicity