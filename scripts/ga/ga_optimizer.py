import json
import numpy as np
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

POPULATION_SIZE = 100
ELITE_SIZE = 15
MUTATION_RATE = 0.12
MUTATION_STRENGTH = 0.35
GENERATIONS = 250
CHECKPOINT_RADIUS = 8.0
END_POINT_TOLERANCE = 1.0   

SEGMENT_FOLDER = "genetic_data/records/SimpleTrack/segments"
CHECKPOINTS_FILE = "assets/SimpleTrack/checkpoints.json"
OBSTACLES_OBJ = "assets/SimpleTrack/Obstacles.obj"
OUTPUT_FOLDER = "genetic_data/best_segments/SimpleTrack"
METADATA_PATH = Path("assets/SimpleTrack/track_metadata.json")


Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

WALL_POLYGONS = None


def load_checkpoints():
    with open(CHECKPOINTS_FILE) as f:
        data = json.load(f)
    cps = sorted(data, key=lambda x: x["id"])
    return np.array([cp["position"] for cp in cps])


checkpoints = load_checkpoints()


# .obj file with walls and obstacles
def parse_obstacles_obj(obj_path):
    global WALL_POLYGONS
    if WALL_POLYGONS is not None:
        return WALL_POLYGONS

    print(f"Parsing {obj_path} + APPLYING TRACK SCALE FROM track_metadata.json...")

    scale = 1.0
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        origin_scale = meta.get("origin_scale", [1, 1, 1])
        scale = origin_scale[0]
        print(f"   → Track scale detected: {scale:.3f} (applied to walls)")
    else:
        print("   → No track_metadata.json found → using scale = 1.0")

    vertices = []
    faces = []

    with open(obj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                x = float(parts[1]) * scale
                z = float(parts[3]) * scale
                vertices.append([-x, z])
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                if len(face) >= 3:
                    faces.append(face)

    polygons = []
    for face in faces:
        poly = np.array([vertices[i] for i in face])
        if np.ptp(poly, axis=0).max() > 4.0:
            polygons.append(poly)

    WALL_POLYGONS = polygons
    print(f"   → Loaded {len(polygons)} scaled & mirrored walls — NOW PERFECT SIZE!")
    return polygons


wall_polygons = parse_obstacles_obj(OBSTACLES_OBJ)


def load_segment(path):
    with open(path) as f:
        frames = json.load(f)
    controls = np.array([f["input"] for f in frames], dtype=np.float32)
    positions = np.array([json.loads(f["position"]) for f in frames])
    speeds = np.array([f["speed"] for f in frames])
    angles = np.array([f["angle"] for f in frames])
    
    seg_id = int(path.stem.split("_")[1])
    next_cp_idx = seg_id + 1
    target_cp = checkpoints[next_cp_idx % len(checkpoints)]

    return {
        "controls": controls,
        "positions": positions,
        "speeds": speeds,
        "angles": angles,
        "target_checkpoint": target_cp,
        "human_end_pos": positions[-1].copy(),  # ← NEW: remember exact human end
        "length": len(frames)
    }

# simulation that mimics car physics (gonna try to simulate it in the track directly)
def simulate(individual, ref):
    pos = ref["positions"][0].copy()
    angle = ref["angles"][0]
    speed = ref["speeds"][0] if len(ref["speeds"]) > 0 else 0.0
    path = [pos.copy()]
    dt = 0.027

    for action in individual:
        w, s, a, d = action

        # FORCE SHARP INPUTS — this is the magic
        throttle = 1.0 if w > 0.5 else 0.0
        brake = 1.0 if s > 0.5 else 0.0
        left = 1.0 if a > 0.5 else 0.0
        right = 1.0 if d > 0.5 else 0.0

        accel = (throttle - brake) * 28
        steer = (right - left) * 9.5

        speed += accel * dt
        speed = np.clip(speed, -10, 52)

        if abs(speed) > 0.5:
            radius = max(1.8, pow(abs(speed)/50, 1.45) * 25 + 1.5)
            turn = (speed * dt / radius) * np.sign(steer) * 57.3
            angle += turn

        dx = math.sin(math.radians(angle)) * speed * dt
        dz = math.cos(math.radians(angle)) * speed * dt
        pos[0] += dx
        pos[2] += dz
        path.append(pos.copy())

    return np.array(path)


def fitness(individual, ref):
    path = simulate(individual, ref)
    target_cp = ref["target_checkpoint"]
    human_end = ref["human_end_pos"]

    # endpoint should be close to the original endpoint
    end_error = np.linalg.norm(path[-1][:2] - human_end[:2])
    if end_error > END_POINT_TOLERANCE:
        return -1e12 - end_error * 10000, end_error, 99.0

    # see if we passed the checkpoint
    min_dist_to_cp = np.min(np.linalg.norm(path[:, :2] - target_cp[:2], axis=1))
    passed_bonus = 5000 * max(0, 1 - min_dist_to_cp / CHECKPOINT_RADIUS)

    deltas = np.diff(path, axis=0)
    dist_traveled = np.sum(np.linalg.norm(deltas, axis=1))

    # deviation from original path
    ref_pos = ref["positions"]
    dists = np.min(np.linalg.norm(ref_pos[None, :, :2] - path[:, None, :2], axis=-1), axis=1)
    avg_dev = np.mean(dists)

    angles = np.arctan2(deltas[:, 2], deltas[:, 0])
    smoothness = np.sum(np.abs(np.diff(angles)))

    score = (
        dist_traveled * 18
        + passed_bonus
        - avg_dev * 70
        - end_error * 500   
        - smoothness * 12
    )

    return score, end_error, min_dist_to_cp

def crossover(p1, p2):
    if len(p1) < 3: return p1.copy(), p2.copy()
    pt = random.randint(1, len(p1)-2)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]


def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            ind[i] += np.random.normal(0, MUTATION_STRENGTH, 4)
            ind[i] = np.clip(ind[i], 0, 1)
    return ind


# plot on the track
def plot(human_path, ga_path, seg_id, track_name, target_cp):
    plt.figure(figsize=(16, 12))

    # === ULTRA TIGHT ZOOM ON SEGMENT ACTION ONLY ===
    all_x = np.concatenate([human_path[:,0], ga_path[:,0]])
    all_z = np.concatenate([human_path[:,2], ga_path[:,2]])
    
    # Super tight margin — only 20-30 meters around the paths
    margin = 25  # You can go down to 15 if you want even tighter!
    plt.xlim(all_x.min() - margin, all_x.max() + margin)
    plt.ylim(all_z.min() - margin, all_z.max() + margin)

    # Only draw walls that are actually near this segment
    center_x, center_z = all_x.mean(), all_z.mean()
    for poly in wall_polygons:
        if (abs(poly[:,0].mean() - center_x) < margin + 50 and 
            abs(poly[:,1].mean() - center_z) < margin + 50):
            patch = Polygon(poly, closed=True, facecolor='#2a2a2a', edgecolor='#888888', alpha=0.85, linewidth=1.2)
            plt.gca().add_patch(patch)

    # === CHECKPOINTS: only show the target (huge) and maybe previous/next if close ===
    current_cp_id = (int(seg_id) + 1) % len(checkpoints)
    prev_cp_id = int(seg_id) % len(checkpoints)
    next_next_id = (int(seg_id) + 2) % len(checkpoints)

    for i, cp_pos in enumerate(checkpoints):
        x, z = cp_pos[0], cp_pos[2]
        dist_to_center = math.hypot(x - center_x, z - center_z)
        
        if i == current_cp_id:
            # BIG target star
            plt.plot(x, z, 'cyan', marker='*', markersize=36, markeredgecolor='black', markeredgewidth=3.5, label="Next Checkpoint", zorder=10)
        elif i in (prev_cp_id, next_next_id) and dist_to_center < margin + 60:
            # Show previous/next only if they're close
            plt.plot(x, z, 'cyan', marker='*', markersize=20, alpha=0.8, zorder=9)
        # Others = invisible (too far)

    # Paths — thick and bold
    plt.plot(human_path[:,0], human_path[:,2], '#44AAFF', linewidth=5.5, label="Human Driver", zorder=4)
    plt.plot(ga_path[:,0], ga_path[:,2], '#FF3333', linewidth=7, label="GA Optimized", zorder=5)

    # Start & GA End
    plt.plot(human_path[0,0], human_path[0,2], 'lime', marker='o', markersize=22, markeredgecolor='black', markeredgewidth=3, label="Start", zorder=11)
    plt.plot(ga_path[-1,0], ga_path[-1,2], '#FF1111', marker='*', markersize=26, markeredgecolor='black', markeredgewidth=3, label="GA End", zorder=11)

    min_dist = np.min(np.linalg.norm(ga_path[:, :2] - target_cp[:2], axis=1))

    plt.title(f"{track_name.upper()} – Segment {seg_id} [ULTRA ZOOM]\n"
              f"GA hits checkpoint at {min_dist:.3f}m — Pure racing perfection",
              fontsize=21, fontweight='bold', color='white', pad=30)

    plt.legend(fontsize=15, loc="upper left", framealpha=0.92)
    plt.grid(alpha=0.3, color='gray')
    ax = plt.gca()
    ax.set_facecolor('#0f0f0f')
    ax.tick_params(colors='white', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.xlabel("X position", color='white', fontsize=15)
    plt.ylabel("Z position", color='white', fontsize=15)
    plt.axis('equal')
    plt.tight_layout()

    plot_file = Path(OUTPUT_FOLDER) / f"plot_segment_{seg_id}.png"
    plt.savefig(plot_file, dpi=400, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()
    print(f"   → ULTRA-ZOOMED GOD TIER PLOT SAVED: {plot_file.name}")



# main loop
segment_paths = Path(SEGMENT_FOLDER).glob("segment_*.json")
segments = sorted(segment_paths, key=lambda p: int(p.stem.split("_")[1]))

track_name = Path(SEGMENT_FOLDER).parent.parent.name

print(f"Starting GA optimization for {track_name.upper()}")
print(f"Found {len(segments)} segments | {len(wall_polygons)} wall polygons loaded\n")

for seg_file in segments:
    sid = seg_file.stem.split("_")[1]
    print(f"{'='*30} OPTIMIZING SEGMENT {sid} → CHECKPOINT {int(sid)+1} {'='*30}")

    ref = load_segment(seg_file)

    pop = [ref["controls"].copy() for _ in range(POPULATION_SIZE)]
    for ind in pop:
        ind += np.random.normal(0, 0.08, ind.shape)
        ind = np.clip(ind, 0, 1)

    best_score = -1e9
    best_ind = pop[0].copy()

    for gen in range(GENERATIONS):
        scored = []
        for ind in pop:
            s, err, min_d = fitness(ind.tolist(), ref)
            scored.append((s, err, min_d, ind))

        scored.sort(key=lambda x: x[0], reverse=True)
        elite = [x[3] for x in scored[:ELITE_SIZE]]

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_ind = scored[0][3].copy()

        if gen % 80 == 0 or gen == GENERATIONS - 1:
            end_err = fitness(best_ind.tolist(), ref)[1]
            print(f"  Gen {gen:3d} → Score {best_score:8.1f} | End error: {end_err:.4f}m")

        new_pop = elite[:]
        while len(new_pop) < POPULATION_SIZE:
            a, b = random.sample(elite, 2)
            c1, c2 = crossover(a.tolist(), b.tolist())
            new_pop.append(np.array(mutate(c1)))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(np.array(mutate(c2)))
        pop = new_pop

    final_path = simulate(best_ind.tolist(), ref)
    final_end_error = np.linalg.norm(final_path[-1][:2] - ref["human_end_pos"][:2])

    out = Path(OUTPUT_FOLDER) / f"best_segment_{sid}.json"
    with open(out, "w") as f:
        json.dump(best_ind.tolist(), f, indent=2)

    plot(ref["positions"], final_path, sid, track_name, ref["target_checkpoint"])

    print(f"→ SEGMENT {sid} COMPLETE | Final end error: {final_end_error:.4f}m |\n")
