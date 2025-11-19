import timeit
import time as real_time
import shutil

import setuptools
from ursina import *
from ursina import curve
from .particles import Particles, TrailRenderer
from .checkpoints import CheckpointHandler
from .sensing_message import SensingSnapshot
from math import pow, atan2
import json
import os

sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
Text.default_resolution = 1080 * Text.size

#FPS = 50
delta_t = 0.01


class Car(Entity):
    def __init__(
        self,
        position=(0, 0, 4),
        rotation=(0, 0, 0),
        topspeed=30,
        acceleration=0.15,
        braking_strength=30,
        friction=1.5,
        camera_speed=8,
    ):
        super().__init__(
            model="assets/cars/sports-car.obj",
            texture="assets/cars/garage/sports-car/sports-red.png",
            collider="sphere",
            position=position,
            rotation=rotation,
        )

        # Controls
        self.controls = "wasd"

        # Car's values
        self.speed = 0
        self.velocity_y = 0
        self.rotation_speed = 0
        self.max_rotation_speed = 1.6
        self.steering_amount = 8
        self.topspeed = topspeed
        self.braking_strenth = braking_strength
        self.camera_speed = camera_speed
        self.acceleration = acceleration
        self.friction = friction
        self.collision_counter = 0
        self.turning_speed = 5
        self.pivot_rotation_distance = 1

        self.reset_position = (0, 0, 0)
        self.reset_rotation = (0, 0, 0)

        # Camera Follow
        self.camera_angle = "top"
        self.camera_offset = (
            0,
            30,
            -35,
        )  # <-- Stuff to change to change the camera distance
        self.camera_rotation = 40
        self.camera_follow = False
        self.change_camera = False
        self.c_pivot = Entity()
        self.camera_pivot = Entity(parent=self.c_pivot, position=self.camera_offset)

        # Pivot for drifting
        self.pivot = Entity()
        self.pivot.position = self.position
        self.pivot.rotation = self.rotation
        self.drifting = False

        # Car Type
        self.car_type = "sports"

        # Particles
        self.particle_time = 0
        self.particle_amount = 0.07  # The lower, the more
        self.particle_pivot = Entity(parent=self)
        self.particle_pivot.position = (0, -1, -2)

        # TrailRenderer
        self.trail_pivot = Entity(parent=self, position=(0, -1, 2))

        self.trail_renderer1 = TrailRenderer(
            parent=self.particle_pivot,
            position=(0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trail_renderer2 = TrailRenderer(
            parent=self.particle_pivot,
            position=(-0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trail_renderer3 = TrailRenderer(
            parent=self.trail_pivot,
            position=(0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trail_renderer4 = TrailRenderer(
            parent=self.trail_pivot,
            position=(-0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )

        self.trails = [
            self.trail_renderer1,
            self.trail_renderer2,
            self.trail_renderer3,
            self.trail_renderer4,
        ]
        self.start_trail = True

        # Collision
        self.copy_normals = False
        self.hitting_wall = False

        self.track = None

        # Graphics
        self.graphics = "fancy"

        # Stopwatch/Timer
        self.timer_running = False
        self.count = 0.0

        self.last_count = self.count
        self.reset_count = 0.0
        self.timer = Text(
            text="", origin=(0, 0), size=0.05, scale=(1, 1), position=(-0.7, 0.43)
        )

        self.reset_count_timer = Text(
            text=str(round(self.reset_count, 1)),
            origin=(0, 0),
            size=0.05,
            scale=(1, 1),
            position=(-0.7, 0.43),
        )

        self.timer.disable()

        self.reset_count_timer.disable()

        self.gamemode = "race"
        self.start_time = False
        self.laps = 0
        self.laps_hs = 0
        self.anti_cheat = 1

        # Bools
        self.driving = False
        self.braking = False

        # Multiplayer
        self.multiplayer = False
        self.multiplayer_update = False
        self.server_running = False

        # Shows whether you are connected to a server or not
        self.connected_text = True
        self.disconnected_text = True

        # Camera shake
        self.shake_amount = 0.1
        self.can_shake = False
        self.camera_shake_option = True

        self.username_text = "Username"

        self.model_path = str(self.model).replace("render/scene/car/", "")

        invoke(self.update_model_path, delay=1)

        self.multiray_sensor = None
        self.checkpoint_handler = None

        # Key press flags for single press detection
        self.c_pressed = False
        self.x_pressed = False
        self.z_pressed = False
        self.g_pressed = False
        self.tab_pressed = False
        self.r_pressed = False

        # Recording
        self.recording = False
        self.recorded_frames = []
        self.frame_idx = 0
        self.recording_start_time = 0.0
        self.last_real_time = real_time.time()

        # Genetic attributes
        self.is_genetic_car = False
        self.genetic_generation = 0
        self.genetic_individual = 0
        self.genetic_segment = 0

        # Checkpoint mode
        self.checkpoint_mode = False
        self.mode_text = Text(
            text="Checkpoint Mode",
            position=(0, -0.35),
            scale=0.8,
            color=color.yellow,
            origin=(0, 0),  # Center it
        )
        self.mode_text.disable()

    def set_track(self, track):
        self.track = track
        self.reset_position = track.car_default_reset_position
        self.reset_orientation = track.car_default_reset_orientation
        self.position = self.reset_position
        self.rotation_y = self.reset_orientation[1]

        # Initialize checkpoint handler
        self.checkpoint_handler = CheckpointHandler(
            track.track_name, track.origin_scale, checkpoint_size=30
        )
        self.checkpoint_handler.hide_ui()

        # Disable rays by default
        if self.multiray_sensor:
            self.multiray_sensor.set_enabled_rays(False)

        # Start recording if genetic car
        if self.is_genetic_car:
            self.start_record()

    def sports_car(self):
        self.car_type = "sports"
        self.model = "assets/cars/sports-car.obj"
        self.texture = "assets/cars/garage/sports-car/sports-red.png"
        self.topspeed = 50
        self.minspeed = -15
        self.acceleration = 25
        self.braking_strenth = 50
        self.turning_speed = 6
        self.max_rotation_speed = 1.6
        self.steering_amount = 9
        self.particle_pivot.position = (0, -1, -1.5)
        self.trail_pivot.position = (0, -1, 1.5)

    def update_camera(self):
        if self.camera_follow:
            if self.change_camera:
                camera.rotation_x = 35
                self.camera_rotation = 40
            self.camera_offset = (0, 60, -70)
            self.camera_speed = 4
            self.change_camera = False
            # camera.rotation_x = self.camera_rotation
            camera.world_position = self.camera_pivot.world_position
            camera.world_rotation_y = self.world_rotation_y

    def check_respawn(self):
        # Respawn
        if held_keys["g"]:
            self.reset_car()

        if held_keys["v"]:
            self.multiray_sensor.set_enabled_rays(not self.multiray_sensor.enabled)

        # Reset the car's position if y value is less than -100
        if self.y <= -100:
            self.reset_car()

        # Reset the car's position if y value is greater than 300
        if self.y >= 300:
            self.reset_car()

    def display_particles(self):
        # Particles

        # Removed particles
        return
        self.particle_time += delta_t
        if self.particle_time >= self.particle_amount:
            self.particle_time = 0
            self.particles = Particles(
                self, self.particle_pivot.world_position - (0, 1, 0)
            )
            self.particles.destroy(1)

    def hand_brake(self):
        # Hand Braking
        if held_keys["space"]:
            if self.rotation_speed < 0:
                self.rotation_speed -= 3 * delta_t
            elif self.rotation_speed > 0:
                self.rotation_speed += 3 * delta_t
            self.speed -= 20 * delta_t

    def compute_steering(self):
        # Steering
        self.rotation_y += self.rotation_speed * 50 * delta_t

        # The car's linear momentum decreases the rotation.
        if self.rotation_speed > 0:
            self.rotation_speed -= self.speed / 6 * delta_t
        elif self.rotation_speed < 0:
            self.rotation_speed += self.speed / 6 * delta_t

        # Can only turn if |speed| > 0.5
        if self.speed > 0.5 or self.speed < -0.5:
            if held_keys[self.controls[1]] or held_keys["left arrow"]:
                self.rotation_speed -= self.steering_amount * delta_t

                # Turning decreases our speed.
                if self.speed > 1:
                    self.speed -= self.turning_speed * delta_t
                elif self.speed < 0:
                    self.speed += self.turning_speed / 5 * delta_t

            elif held_keys[self.controls[3]] or held_keys["right arrow"]:
                self.rotation_speed += self.steering_amount * delta_t
                if self.speed > 1:
                    self.speed -= self.turning_speed * delta_t
                elif self.speed < 0:
                    self.speed += self.turning_speed / 5 * delta_t
            # If no keys pressed, the rotation speed goes down.
            else:
                if self.rotation_speed > 0:
                    self.rotation_speed -= 5 * delta_t
                elif self.rotation_speed < 0:
                    self.rotation_speed += 5 * delta_t
        else:
            self.rotation_speed = 0

    def cap_kinetic_parameters(self):
        # Cap the speed
        if self.speed >= self.topspeed:
            self.speed = self.topspeed
        if self.speed <= -15:
            self.speed = -15
        if self.speed <= 0:
            self.pivot.rotation_y = self.rotation_y

        # Cap the steering
        if self.rotation_speed >= self.max_rotation_speed:
            self.rotation_speed = self.max_rotation_speed
        if self.rotation_speed <= -self.max_rotation_speed:
            self.rotation_speed = -self.max_rotation_speed

        # Cap the camera rotation
        if self.camera_rotation >= 40:
            self.camera_rotation = 40
        elif self.camera_rotation <= 30:
            self.camera_rotation = 30

    def update_vertical_position(self, y_ray, movementY):
        # Check if car is hitting the ground
        if self.visible:
            if y_ray.distance <= self.scale_y * 1.7 + abs(movementY):
                self.velocity_y = 0
                # Check if hitting a wall or steep slope
                if (
                    y_ray.world_normal.y > 0.7
                    and y_ray.world_point.y - self.world_y < 0.5
                ):
                    # Set the y value to the ground's y value
                    self.y = y_ray.world_point.y + 1.4
                    self.hitting_wall = False
                else:
                    # Car is hitting a wall
                    self.hitting_wall = True

                if self.copy_normals:
                    self.ground_normal = self.position + y_ray.world_normal
                else:
                    self.ground_normal = self.position + (0, 180, 0)
            else:
                self.y += movementY * 50 * delta_t
                self.velocity_y -= 50 * delta_t

    def update(self):
        if hasattr(self, 'follow_path') and self.follow_path:
            return
        dt_real = real_time.time() - self.last_real_time
        self.count += dt_real
        self.last_real_time = real_time.time()
        # Exit if esc pressed.
        if held_keys["escape"]:
            quit()

        self.check_respawn()

        # Toggle checkpoint mode with Tab
        if held_keys["tab"] and not self.tab_pressed:
            self.tab_pressed = True
            self.checkpoint_mode = not self.checkpoint_mode
            if self.checkpoint_mode:
                self.checkpoint_handler.ui_enabled = True
                self.checkpoint_handler.show_ui()
                if self.multiray_sensor:
                    self.multiray_sensor.set_enabled_rays(True)
                self.mode_text.enable()
                print("Checkpoint mode enabled")
            else:
                self.checkpoint_handler.ui_enabled = False
                self.checkpoint_handler.hide_ui()
                if self.multiray_sensor:
                    self.multiray_sensor.set_enabled_rays(False)
                self.mode_text.disable()
                print("Checkpoint mode disabled")
        elif not held_keys["tab"]:
            self.tab_pressed = False

        # Handle checkpoint placement
        if self.checkpoint_handler and self.checkpoint_mode:
            if held_keys["c"] and not self.c_pressed:  # Place checkpoint
                self.c_pressed = True
                self.checkpoint_handler.place_checkpoint(
                    self.position, (0, self.rotation_y, 0), self.multiray_sensor
                )
            elif not held_keys["c"]:
                self.c_pressed = False

            if held_keys["x"] and not self.x_pressed:  # Remove last checkpoint
                self.x_pressed = True
                self.checkpoint_handler.remove_last_checkpoint()
            elif not held_keys["x"]:
                self.x_pressed = False

            if held_keys["z"] and not self.z_pressed:  # Clear all checkpoints
                self.z_pressed = True
                self.checkpoint_handler.clear_all_checkpoints()
            elif not held_keys["z"]:
                self.z_pressed = False

        if held_keys["g"] and not self.g_pressed:  # Reset car and checkpoint counts
            self.g_pressed = True
            self.reset_car()
            if self.checkpoint_handler:
                self.checkpoint_handler.current_lap = 0
                self.checkpoint_handler.next_checkpoint_index = 0
                self.checkpoint_handler.passed_checkpoints.clear()
                for entity_data in self.checkpoint_handler.checkpoint_entities:
                    entity_data["passed"] = False
                    entity_data["entity"].color = color.green
                    entity_data["text"].color = color.yellow
                print("Car and checkpoint counts reset")
        elif not held_keys["g"]:
            self.g_pressed = False

        # Recording toggle
        if held_keys["r"] and not self.r_pressed:
            self.r_pressed = True
            if not self.recording:
                self.start_record()
            else:
                self.stop_record()
        elif not held_keys["r"]:
            self.r_pressed = False

        #   Process inputs & update speed
        if held_keys[self.controls[0]] or held_keys["up arrow"]:
            self.speed += self.acceleration * delta_t
            self.driving = True

            self.display_particles()
        else:
            self.driving = False
            if self.speed > 1:
                self.speed -= self.friction * 5 * delta_t
            elif self.speed < -1:
                self.speed += self.friction * 5 * delta_t

        # Braking
        if held_keys[self.controls[2] or held_keys["down arrow"]]:
            if self.speed > 0:
                self.speed -= self.braking_strenth * delta_t
            else:
                self.speed -= self.acceleration * delta_t
            self.braking = True
        else:
            self.braking = False

        #   Check physical constrains
        if self.speed > self.topspeed:
            self.speed = self.topspeed
        elif self.speed < self.minspeed:
            self.speed = self.minspeed

        if (
            held_keys[self.controls[1]]
            or held_keys["left arrow"]
            or held_keys[self.controls[3]]
            or held_keys["right arrow"]
        ):
            turn_right = held_keys[self.controls[3]] or held_keys["right arrow"]
            rotation_sign = 1 if turn_right else -1

            #   Max angular speed
            normalized_speed = abs(self.speed / self.topspeed)

            #   function to map unit speed (between 0 and max speed) to a rotation coefficient space.
            #   Rotation radius is function of speed
            def rotation_radius(normalized_speed):
                smallest_radius = 1.5
                biggest_radius = 25
                return (
                    pow(normalized_speed, 1.5) * (biggest_radius - smallest_radius)
                    + smallest_radius
                )

            #   Get rotation radius
            radius = rotation_radius(normalized_speed)

            #   Get travelled distance
            travelled_dist = abs(self.speed * delta_t)
            #   Project on circle radius & compute angle variation seen from the center of the circle
            travelled_circle_center_angle = travelled_dist / radius
            #   Compute variation in Y & X
            dx = 1 - cos(travelled_circle_center_angle)
            dy = sin(travelled_circle_center_angle)

            da = atan2(dx, dy) / 3.14159 * 180

            self.rotation_y += da * rotation_sign

        #   Integrate speed into movement
        total_dist_to_move = self.speed * delta_t

        #   Check collision via recast

        #   Return residual distance to travel and residual speed.
        def move_car(distance_to_travel, direction):
            front_collision = boxcast(
                origin=self.world_position,
                direction=self.forward * direction,
                thickness=(0.1, 0.1),
                distance=self.scale_x + distance_to_travel,
                ignore=[
                    self,
                ],
            )

            #   Detect collision
            if front_collision.distance < self.scale_x + distance_to_travel:
                self.collision_counter += 1
                free_dist = front_collision.distance - self.scale_x + distance_to_travel

                #   cancel speed going directly into the obstacle
                next_forward = (
                    self.forward
                    - (self.forward.dot(front_collision.world_normal))
                    * front_collision.world_normal
                )
                self.speed = self.speed * (
                    0.5 + 0.5 * (self.forward.dot(front_collision.world_normal))
                )  # Loose half speed on collision and some depending on the angle

                self.rotation_y = (
                    atan2(next_forward[0], next_forward[2]) / 3.14159 * 180
                )
                dist_left_to_travel = distance_to_travel - free_dist

                #   Move car away from obstacle to prevent overlap due to *¦@+!? physics system
                OBSTACLE_DISPLACEMENT_MARGIN = 1
                self.x += (
                    front_collision.world_normal * OBSTACLE_DISPLACEMENT_MARGIN
                ).x
                self.z += (
                    front_collision.world_normal * OBSTACLE_DISPLACEMENT_MARGIN
                ).z

                return 0

            else:
                self.x += self.forward[0] * distance_to_travel
                self.z += self.forward[2] * distance_to_travel

                return 0

        for i in range(2):
            total_dist_to_move = move_car(
                total_dist_to_move, 1 if self.speed > 0 else -1
            )

            if total_dist_to_move <= 0:
                break

        self.c_pivot.position = self.position
        self.c_pivot.rotation_y = self.rotation_y
        self.update_camera()

        self.pivot.position = self.position

        # Update checkpoint handler
        if self.checkpoint_handler and not getattr(self, 'disable_checkpoint_check', False):
            self.checkpoint_handler.update()
            self.checkpoint_handler.check_passed_checkpoints(self.position)

        # Update autopilot if present
        if hasattr(self, "autopilot") and self.autopilot:
            snapshot = SensingSnapshot()
            snapshot.current_controls = (
                held_keys["w"] or held_keys["up arrow"],
                held_keys["s"] or held_keys["down arrow"],
                held_keys["a"] or held_keys["left arrow"],
                held_keys["d"] or held_keys["right arrow"],
            )
            snapshot.car_position = self.world_position
            snapshot.car_speed = self.speed
            snapshot.car_angle = self.rotation_y
            snapshot.raycast_distances = (
                self.multiray_sensor.collect_sensor_values()
                if self.multiray_sensor
                else []
            )
            snapshot.collision_counter = self.collision_counter
            if self.checkpoint_handler:
                snapshot.checkpoints_passed = len(
                    self.checkpoint_handler.passed_checkpoints
                )
                snapshot.total_checkpoints = len(
                    self.checkpoint_handler.lap_checkpoints
                )
            else:
                snapshot.checkpoints_passed = 0
                snapshot.total_checkpoints = 0
            self.autopilot.update(snapshot)

        # Record keys if recording
        if self.recording:
            self.save_frame()

    def reset_car(self):
        """
        Resets the car
        """
        #   Project car directly on ground when resetting
        self.position = self.reset_position
        # y_ray = raycast(origin = self.reset_position, direction = (0,-1,0), ignore = [self,])
        # self.y = y_ray.world_point.y + 1.4
        print(self.reset_orientation)
        self.rotation_y = self.reset_orientation[1]

        print("reseting at", str(self.position), " --> ", self.rotation_y)

        camera.world_rotation_y = self.rotation_y
        self.speed = 0
        self.velocity_y = 0
        self.timer_running = False
        for trail in self.trails:
            if trail.trailing:
                trail.end_trail()
        self.start_trail = True

    def simple_intersects(self, entity):
        """
        A faster AABB intersects for detecting collision with
        simple objects, doesn't take rotation into account
        """
        minXA = self.x - self.scale_x
        maxXA = self.x + self.scale_x
        minYA = self.y - self.scale_y + (self.scale_y / 2)
        maxYA = self.y + self.scale_y - (self.scale_y / 2)
        minZA = self.z - self.scale_z
        maxZA = self.z + self.scale_z

        minXB = entity.x - entity.scale_x + (entity.scale_x / 2)
        maxXB = entity.x + entity.scale_x - (entity.scale_x / 2)
        minYB = entity.y - entity.scale_y + (entity.scale_y / 2)
        maxYB = entity.y + entity.scale_y - (entity.scale_y / 2)
        minZB = entity.z - entity.scale_z + (entity.scale_z / 2)
        maxZB = entity.z + entity.scale_z - (entity.scale_z / 2)

        return (
            (minXA <= maxXB and maxXA >= minXB)
            and (minYA <= maxYB and maxYA >= minYB)
            and (minZA <= maxZB and maxZA >= minZB)
        )

    def reset_timer(self):
        """
        Resets the timer
        """
        self.count = self.reset_count
        self.timer.enable()
        self.reset_count_timer.disable()

    def reset_collision_counter(self):
        """
        Resets the collision counter
        """
        self.collision_counter = 0

    def start_record(self):
        """
        Starts recording manual records
        """
        self.recording = True
        trackname = self.track.track_name if self.track else "unknown"
        path = f"genetic_data/records/{trackname}/"
        # Never clear genetic_data/records to preserve data for GA initialization
        os.makedirs(path, exist_ok=True)
        self.recorded_frames = []
        self.frame_idx = 0
        self.recording_start_time = real_time.time()
        print("Recording started")

    def stop_record(self):
        """
        Stops recording and saves the records
        """
        self.recording = False
        print("Recording stopped")
        print(f"is_genetic_car: {self.is_genetic_car}, recorded_frames: {len(self.recorded_frames)}")
        if self.is_genetic_car:
            print(f"Genetic attributes: gen={self.genetic_generation}, ind={self.genetic_individual}, seg={self.genetic_segment}")
            # Save to genetic path
            dir_path = f"genetic_data/populations/{self.track.track_name if self.track else 'unknown'}/segment_{self.genetic_segment}/generation_{self.genetic_generation}"
            os.makedirs(dir_path, exist_ok=True)
            file_path = f"{dir_path}/individual_{self.genetic_individual}.json"
            with open(file_path, "w") as f:
                json.dump(self.recorded_frames, f)
            print(
                f"Genetic individual {self.genetic_individual} record saved to {file_path}"
            )
        else:
            # Save recorded data
            trackname = self.track.track_name if self.track else "unknown"
            path = f"genetic_data/records/{trackname}/"
            complete_path = f"{path}complete_record.json"
            with open(complete_path, "w") as f:
                json.dump(self.recorded_frames, f)
            print(f"Complete record saved to {complete_path}")

            # Split into segments (only from first lap)
            segments_path = f"{path}segments/"
            os.makedirs(segments_path, exist_ok=True)
            from collections import defaultdict

            checkpoint_to_frames = defaultdict(list)
            for frame in self.recorded_frames:
                if frame.get("lap", -1) == 0:  # Only first lap
                    cp = frame["checkpoint"]
                    checkpoint_to_frames[cp].append(frame)
            segments = sorted(checkpoint_to_frames.keys())
            for i, cp in enumerate(segments):
                segment_data = checkpoint_to_frames[cp]
                segment_file = f"{segments_path}segment_{i}.json"
                with open(segment_file, "w") as f:
                    json.dump(segment_data, f)
                print(f"Segment {i} (checkpoint {cp}) saved to {segment_file}")

    def save_frame(self):
        """
        Saves the current frame to recorded_frames if recording is active
        """
        if not self.recording:
            return
        self.recorded_frames.append(
            {
                "idx": self.frame_idx,
                "time": real_time.time() - self.recording_start_time,
                "input": [
                    int(held_keys["w"] or held_keys["up arrow"]),  # forward
                    int(held_keys["s"] or held_keys["down arrow"]),  # backward
                    int(held_keys["a"] or held_keys["left arrow"]),  # left
                    int(held_keys["d"] or held_keys["right arrow"]),  # right
                ],
                "angle": self.rotation_y,
                "speed": self.speed,
                "position": json.dumps(list(self.position)),
                "checkpoint": (
                    -1
                    if self.checkpoint_handler
                    and len(self.checkpoint_handler.passed_checkpoints) == 0
                    else self.checkpoint_handler.next_checkpoint_index - 1
                    if self.checkpoint_handler
                    else -1
                ),
                "lap": (
                    -1
                    if self.checkpoint_handler
                    and len(self.checkpoint_handler.passed_checkpoints) == 0
                    else self.checkpoint_handler.current_lap
                    if self.checkpoint_handler
                    else -1
                ),
            }
        )
        self.frame_idx += 1

    def animate_text(self, text, top=1.2, bottom=0.6):
        """
        Animates the scale of text
        """
        if self.gamemode != "drift":
            if self.last_count > 1:
                text.animate_scale((top, top, top), curve=curve.out_expo)
                invoke(text.animate_scale, (bottom, bottom, bottom), delay=0.2)
        else:
            text.animate_scale((top, top, top), curve=curve.out_expo)
            invoke(text.animate_scale, (bottom, bottom, bottom), delay=0.2)

    def update_model_path(self):
        """
        Updates the model's file path for multiplayer
        """
        self.model_path = str(self.model).replace("render/scene/car/", "")
        invoke(self.update_model_path, delay=3)


# Class for copying the car's position, rotation for multiplayer
class CarRepresentation(Entity):
    def __init__(self, car, position=(0, 0, 0), rotation=(0, 65, 0)):
        super().__init__(
            parent=scene,
            model="assets/cars/sports-car.obj",
            texture="assets/cars/garage/sports-car/sports-red.png",
            position=position,
            rotation=rotation,
            scale=(1, 1, 1),
        )

        self.model_path = str(self.model).replace(
            "render/scene/car_representation/", ""
        )

        self.text_object = None


# Username shown above the car
class CarUsername(Text):
    def __init__(self, car):
        super().__init__(
            parent=car, text="Guest", y=3, scale=30, color=color.white, billboard=True
        )

        self.username_text = "Guest"

    def update(self):
        self.text = self.username_text