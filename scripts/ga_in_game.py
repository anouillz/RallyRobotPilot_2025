from rallyrobopilot.game_launcher import prepare_game_app
from ursina import *
import json
import numpy as np
import random
import math
from pathlib import Path

# ================= CONFIGURATION =================
SEGMENT_ID = 9                         
POPULATION_SIZE = 100                    
ELITE_SIZE = 15                         
MUTATION_RATE = 0.6                    # Increased for more diversity
MUTATION_STRENGTH = 0.8                # Increased for stronger mutations
GENERATIONS = 30
SEGMENT_FOLDER = "genetic_data/records/SimpleTrack/segments"
OUTPUT_BEST = Path("genetic_data/best_segments/SimpleTrack")
OUTPUT_BEST.mkdir(parents=True, exist_ok=True)
TRACK_NAME = "SimpleTrack"
# ================================================


class GhostCar(Entity):
    def __init__(self, path, ga):
        super().__init__()
        self.path = path
        self.ga = ga
        self.model = 'cube'
        self.scale = (2, 1, 4)
        self.color = color.azure.tint(-.3)
        self.alpha = 0.6

    def update(self):
        step = self.ga.global_step
        if step < len(self.path):
            pos = self.path[int(step)]
            self.position = Vec3(*pos)
        else:
            # Loop
            pos = self.path[int(step) % len(self.path)]
            self.position = Vec3(*pos)


class GAInGame:
    def __init__(self, car):
        self.car = car
        self.segment_id = SEGMENT_ID
        self.track_name = TRACK_NAME
        self.ref = self.load_reference()
        self.checkpoints = self.load_checkpoints()
        self.population = self.create_population()
        self.generation = 0
        self.best_score = -1e20
        self.best_line = Entity(color=color.lime)  # Create empty entity without model
        self.ghost_cars = []  # Track ghost cars to destroy them later
        self.global_step = 0.0  # Global step for synchronization

        # UI Text for generation
        self.gen_text = Text(text=f"GEN {self.generation}", position=(0.5, -0.4), scale=1, color=color.black)

        # Set car to segment start position and angle
        self.car.position = Vec3(*self.ref["positions"][0])
        self.car.rotation_y = self.ref["angles"][0]


        print(f"GA IN-GAME STARTED — Segment {self.segment_id}")

    def load_reference(self):
        file = Path(SEGMENT_FOLDER) / f"segment_{self.segment_id}.json"
        with open(file) as f:
            frames = json.load(f)
        controls = np.array([f["input"] for f in frames], dtype=np.float64)
        positions = np.array([json.loads(f["position"]) for f in frames])
        speeds = np.array([f["speed"] for f in frames])
        angles = np.array([f["angle"] for f in frames])
        return {"controls": controls, "positions": positions, "speeds": speeds, "angles": angles}

    def load_checkpoints(self):
        file = Path(f"assets/{self.track_name}/checkpoints.json")
        if file.exists():
            with open(file) as f:
                cps = json.load(f)
                for cp in cps:
                    cp['passed'] = False
                return cps
        else:
            return []

    def create_population(self):
        pop = [self.ref["controls"].copy() for _ in range(POPULATION_SIZE)]
        for ind in pop:
            ind += np.random.normal(0, 0.3, ind.shape)  # Increased initial noise
            np.clip(ind, 0, 1, out=ind)
        return pop

    def crossover(self, p1, p2):
        if len(p1) < 3: return p1.copy(), p2.copy()
        pt = random.randint(1, len(p1)-2)
        return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])

    def mutate(self, ind):
        ind = ind.copy()
        mask = np.random.random(ind.shape) < MUTATION_RATE
        ind[mask] += np.random.normal(0, MUTATION_STRENGTH, ind.shape)[mask]
        np.clip(ind, 0, 1, out=ind)
        return ind

    def _did_cross_checkpoint(self, prev_pos, curr_pos, checkpoint):
        prev = tuple(prev_pos) if hasattr(prev_pos, "__iter__") else prev_pos
        curr = tuple(curr_pos) if hasattr(curr_pos, "__iter__") else curr_pos
        cp = tuple(checkpoint["position"])
        rot_y = checkpoint["rotation"][1] if len(checkpoint["rotation"]) > 1 else 0
        perp_x = math.cos(math.radians(rot_y))
        perp_z = math.sin(math.radians(rot_y))
        forward_x = math.sin(math.radians(rot_y))
        forward_z = math.cos(math.radians(rot_y))
        prev_vec = (prev[0] - cp[0], 0, prev[2] - cp[2])
        curr_vec = (curr[0] - cp[0], 0, curr[2] - cp[2])
        move_vec = (curr[0] - prev[0], 0, curr[2] - prev[2])
        prev_side = prev_vec[0] * forward_x + prev_vec[2] * forward_z
        curr_side = curr_vec[0] * forward_x + curr_vec[2] * forward_z
        move_dot_forward = move_vec[0] * forward_x + move_vec[2] * forward_z
        checkpoint_width = checkpoint.get("size", 25) / 2
        checkpoint_thickness = 0.2
        if prev_side * curr_side >= 0:
            return False
        t = -prev_side / (curr_side - prev_side) if curr_side != prev_side else 0
        t = max(0, min(1, t))
        cross_x = prev[0] + t * (curr[0] - prev[0])
        cross_z = prev[2] + t * (curr[2] - prev[2])
        cross_vec = (cross_x - cp[0], 0, cross_z - cp[2])
        cross_distance_width = abs(cross_vec[0] * perp_x + cross_vec[2] * perp_z)
        cross_distance_length = abs(cross_vec[0] * forward_x + cross_vec[2] * forward_z)
        within_width = cross_distance_width <= checkpoint_width
        within_length = cross_distance_length <= checkpoint_thickness
        moving_towards = move_dot_forward > 0
        return within_width and within_length and moving_towards

    def simulate(self, individual):
        pos = self.ref["positions"][0].copy()
        angle = self.ref["angles"][0]
        speed = self.ref["speeds"][0]
        path = [pos.copy()]
        dt = 0.027
        collision_counter = 0
        has_printed_collision = False

        for i, action in enumerate(individual):
            w, s, a, d = action 
            accel = (w - s) * 28 
            steer = (d - a) * 9.5

            speed += accel * dt
            speed = np.clip(speed, -10, 52)  

            if abs(speed) > 0.5:
                radius = max(1.8, pow(abs(speed)/50, 1.45) * 25 + 1.5)
                turn = (speed * dt / radius) * np.sign(steer) * 57.3
                angle += turn

            dx = math.sin(math.radians(angle)) * speed * dt
            dz = math.cos(math.radians(angle)) * speed * dt

            # Check for collision every 5 steps
            if i % 5 == 0:
                dist = math.sqrt(dx**2 + dz**2)
                if dist > 0.1:  # Only check if moving significantly
                    direction = Vec3(dx/dist, 0, dz/dist)
                    hit = raycast(
                        origin=Vec3(pos[0], pos[1] + 1, pos[2]),  # Start from above to avoid self-collision
                        direction=direction,
                        distance=dist + 1,  # Shorter check
                        ignore=[self.car]  # Ignore the real car
                    )
                    if hit and hit.distance < dist + 1:
                        collision_counter += 1
                        speed *= 0.5  # Penalty for collision
                        if not has_printed_collision:
                            #print("COLLISION")
                            has_printed_collision = True

            pos[0] += dx
            pos[2] += dz
            path.append(pos.copy())

        return np.array(path), collision_counter

    def fitness(self, individual):
        path, collisions = self.simulate(individual)
        if collisions > 0:
            return -1e15

        # Reset passed
        for cp in self.checkpoints:
            cp['passed'] = False

        reached = 0
        prev_pos = path[0]
        for pos in path[1:]:
            for cp in self.checkpoints:
                if not cp['passed']:
                    if self._did_cross_checkpoint(prev_pos, pos, cp):
                        cp['passed'] = True
                        reached += 1
            prev_pos = pos

        total_cp = len(self.checkpoints)
        missing = total_cp - reached
        dist_traveled = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        end_error = np.linalg.norm(path[-1] - self.ref["positions"][-1])
        score = - (dist_traveled + missing * 1000 + end_error * 10)  # Minimize dist, missing, and end error
        return score

    def update(self):
        if self.generation >= GENERATIONS:
            return

        if time.time() - getattr(self, "last_time", 0) < 3.0:
            return
        self.last_time = time.time()

        scored = [(self.fitness(ind.tolist()), ind) for ind in self.population]
        scored.sort(key=lambda x: x[0], reverse=True)
        elite = [x[1] for x in scored[:ELITE_SIZE]]

        is_new_best = scored[0][0] > self.best_score

        if is_new_best:
            self.best_score = scored[0][0]
            self.best_individual = scored[0][1].copy()
            path, collisions = self.simulate(self.best_individual)

            # Calculate reached
            for cp in self.checkpoints:
                cp['passed'] = False
            reached = 0
            prev_pos = path[0]
            for pos in path[1:]:
                for cp in self.checkpoints:
                    if not cp['passed']:
                        if self._did_cross_checkpoint(prev_pos, pos, cp):
                            cp['passed'] = True
                            reached += 1
                prev_pos = pos
            total_cp = len(self.checkpoints)

            # Only update line if path has points
            if len(path) > 1:
                verts = [Vec3(*p) for p in path]
                if not hasattr(self.best_line, 'model') or self.best_line.model is None:
                    self.best_line.model = Mesh(vertices=verts, mode='line', thickness=8)
                else:
                    self.best_line.model.vertices = verts
                    self.best_line.model.generate()

            out = OUTPUT_BEST / f"real_best_segment_{self.segment_id}.json"
            with open(out, "w") as f:
                json.dump(self.best_individual.tolist(), f, indent=2)
            print(f"GEN {self.generation} | NEW BEST | Score: {self.best_score:.1f} | Collisions: {collisions} | Checkpoints: {reached}/{total_cp}")
            self.best_score = scored[0][0]
            self.best_individual = scored[0][1].copy()
            path, collisions = self.simulate(self.best_individual)

            # Calculate reached
            for cp in self.checkpoints:
                cp['passed'] = False
            reached = 0
            prev_pos = path[0]
            for pos in path[1:]:
                for cp in self.checkpoints:
                    if not cp['passed']:
                        if self._did_cross_checkpoint(prev_pos, pos, cp):
                            cp['passed'] = True
                            reached += 1
                prev_pos = pos
            total_cp = len(self.checkpoints)

            # Only update line if path has points
            if len(path) > 1:
                verts = [Vec3(*p) for p in path]
                if not hasattr(self.best_line, 'model') or self.best_line.model is None:
                    self.best_line.model = Mesh(vertices=verts, mode='line', thickness=8)
                else:
                    self.best_line.model.vertices = verts
                    self.best_line.model.generate()

            out = OUTPUT_BEST / f"real_best_segment_{self.segment_id}.json"
            with open(out, "w") as f:
                json.dump(self.best_individual.tolist(), f, indent=2)
            print(f"GEN {self.generation} | NEW BEST | Score: {self.best_score:.1f} | Collisions: {collisions} | Checkpoints: {reached}/{total_cp}")

        # Show ghost cars
        for gc in self.ghost_cars:
            destroy(gc)
        self.ghost_cars = []
        for i, (_, ind) in enumerate(scored[:5]):  # Show only top 5 ghost cars
            path, _ = self.simulate(ind.tolist())
            if len(path) > 1:
                gc = GhostCar(path, self)
                # Make the best individual green
                if i == 0:
                    gc.color = color.green
                self.ghost_cars.append(gc)

        # Breed
        new_pop = elite[:]
        while len(new_pop) < POPULATION_SIZE:
            a, b = random.sample(elite, 2)
            c1, c2 = self.crossover(a, b)
            new_pop.append(self.mutate(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(self.mutate(c2))
        self.population = new_pop
        self.generation += 1
        self.gen_text.text = f"GEN {self.generation}"


if __name__ == "__main__":
    app, car = prepare_game_app("SimpleTrack")
    ga = GAInGame(car)
    #EditorCamera()

    def update():
        ga.update()

        # Increment global step slower for synchronization
        ga.global_step += 0.3 

        # Move the car along the reference path
        idx = int(ga.global_step) % len(ga.ref["positions"])
        ga.car.position = Vec3(*ga.ref["positions"][idx])
        ga.car.rotation_y = ga.ref["angles"][idx]

        ga.car.update_camera()  # Update camera to follow the car

    app.run()

