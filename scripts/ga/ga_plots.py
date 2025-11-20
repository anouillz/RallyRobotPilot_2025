# ga_plots.py — ULTIMATE PLOTS: Trajectory + Frames Saved + Fitness + Deviation
import json
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ================= CONFIGURATION =================
TRACK_NAME = "SimpleTrack"
SEGMENT_FOLDER = f"genetic_data/records/{TRACK_NAME}/segments"
BEST_FOLDER = f"genetic_data/best_segments/{TRACK_NAME}"
STATS_FOLDER = "genetic_data/stats"
OBSTACLES_OBJ = f"assets/{TRACK_NAME}/Obstacles.obj"
CHECKPOINTS_FILE = f"assets/{TRACK_NAME}/checkpoints.json"
METADATA_FILE = f"assets/{TRACK_NAME}/track_metadata.json"
PLOT_FOLDER = Path("plots_ga_analysis")
PLOT_FOLDER.mkdir(exist_ok=True)
# ================================================

# Load checkpoints
with open(CHECKPOINTS_FILE) as f:
    checkpoints = np.array([cp["position"] for cp in sorted(json.load(f), key=lambda x: x["id"])])

# Load scale
scale = 1.0
if Path(METADATA_FILE).exists():
    with open(METADATA_FILE) as f:
        scale = json.load(f).get("origin_scale", [1,1,1])[0]

# Load walls
vertices = []
faces = []
with open(OBSTACLES_OBJ) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        if parts[0] == 'v':
            x = float(parts[1]) * scale
            z = float(parts[3]) * scale
            vertices.append([-x, z])
        elif parts[0] == 'f':
            face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
            if len(face) >= 3:
                faces.append(face)

wall_polygons = []
for face in faces:
    poly = np.array([vertices[i] for i in face])
    if np.ptp(poly, axis=0).max() > 4.0:
        wall_polygons.append(poly)

# Simulation function
def simulate(individual, start_pos, start_angle, start_speed):
    pos = np.array(start_pos)
    angle = start_angle
    speed = start_speed
    path = [pos.copy()]
    dt = 0.027

    for action in individual:
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
        pos[0] += dx
        pos[2] += dz
        path.append(pos.copy())

    return np.array(path)

# Process each segment
best_files = sorted(Path(BEST_FOLDER).glob("best_segment_*.json"))
total_saved_frames = 0

for best_file in best_files:
    seg_id = int(best_file.stem.split("_")[-1])
    human_file = Path(SEGMENT_FOLDER) / f"segment_{seg_id}.json"
    stats_file = Path(STATS_FOLDER) / f"segment_{seg_id}_stats.json"

    if not human_file.exists():
        print(f"Skipping segment {seg_id} — no human reference")
        continue

    # Load human
    with open(human_file) as f:
        human_frames = json.load(f)
    human_pos = np.array([json.loads(f["position"]) for f in human_frames])
    start_pos = human_pos[0]
    start_angle = human_frames[0]["angle"]
    start_speed = human_frames[0]["speed"]
    original_frames = len(human_frames)

    # Load GA best
    with open(best_file) as f:
        ga_controls = json.load(f)
    ga_path = simulate(ga_controls, start_pos, start_angle, start_speed)
    ga_frames = len(ga_controls)

    # Frames & time saved
    frames_saved = original_frames - ga_frames
    total_saved_frames += frames_saved

    # === 1. TRAJECTORY PLOT ===
    plt.figure(figsize=(16, 12))
    all_x = np.concatenate([human_pos[:,0], ga_path[:,0]])
    all_z = np.concatenate([human_pos[:,2], ga_path[:,2]])
    margin = 40
    plt.xlim(all_x.min() - margin, all_x.max() + margin)
    plt.ylim(all_z.min() - margin, all_z.max() + margin)

    center_x, center_z = all_x.mean(), all_z.mean()
    for poly in wall_polygons:
        if (abs(poly[:,0].mean() - center_x) < margin + 80 and 
            abs(poly[:,1].mean() - center_z) < margin + 80):
            plt.gca().add_patch(Polygon(poly, closed=True, facecolor='#2a2a2a', edgecolor='#888888', alpha=0.9, linewidth=1.2))

    plt.plot(human_pos[:,0], human_pos[:,2], '#4488FF', linewidth=5.5, label="Human", zorder=4)
    plt.plot(ga_path[:,0], ga_path[:,2], '#FF3333', linewidth=7.5, label="GA", zorder=5)

    plt.plot(start_pos[0], start_pos[2], 'lime', marker='o', markersize=22, markeredgecolor='black', label="Start", zorder=11)
    plt.plot(ga_path[-1,0], ga_path[-1,2], 'red', marker='*', markersize=28, markeredgecolor='black', label="End", zorder=11)

    plt.title(f"Segment {seg_id}\nFrames saved: {frames_saved}",
              fontsize=21, fontweight='bold', color='white', pad=30)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.25)
    ax = plt.gca()
    ax.set_facecolor('#0f0f0f')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.xlabel("X", color='white')
    plt.ylabel("Z", color='white')
    plt.axis('equal')
    plt.tight_layout()

    plot_file = PLOT_FOLDER / f"traj_segment_{seg_id}.png"
    plt.savefig(plot_file, dpi=400, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()

    # === 2. STATS PLOTS (fitness + deviation) ===
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

        # Fitness evolution
        plt.figure(figsize=(12, 5))
        plt.plot(stats["fitness_history"], color='lime', linewidth=3)
        plt.title(f"Segment {seg_id} — Fitness Evolution", fontsize=18, color='white')
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.grid(alpha=0.3)
        ax = plt.gca()
        ax.set_facecolor('#0f0f0f')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        plt.tight_layout()
        plt.savefig(PLOT_FOLDER / f"fitness_segment_{seg_id}.png", dpi=300, bbox_inches='tight', facecolor='#0f0f0f')
        plt.close()

        # Max deviation
        plt.figure(figsize=(12, 5))
        plt.plot(stats["max_dev_history"], color='orange', linewidth=3)
        plt.title(f"Segment {seg_id} — Max Deviation from Human Line", fontsize=18, color='white')
        plt.xlabel("Generation")
        plt.ylabel("Max Deviation (m)")
        plt.grid(alpha=0.3)
        ax = plt.gca()
        ax.set_facecolor('#0f0f0f')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        plt.tight_layout()
        plt.savefig(PLOT_FOLDER / f"deviation_segment_{seg_id}.png", dpi=300, bbox_inches='tight', facecolor='#0f0f0f')
        plt.close()

print(f"\nALL PLOTS DONE — {len(best_files)} segments")
print(f"TOTAL FRAMES SAVED: {total_saved_frames}")
print(f"Check folder: {PLOT_FOLDER}")