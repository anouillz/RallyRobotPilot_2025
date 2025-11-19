#!/usr/bin/env python
import lzma
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ACTIONS = ["forward", "back", "left", "right"]


def load_controls_from_file(path: Path) -> np.ndarray:
    """Charge tous les current_controls d'un fichier (N, 4)."""
    with lzma.open(path, "rb") as f:
        data = pickle.load(f)

    ctrls = []
    for e in data:
        ctrls.append(np.array(e.current_controls, dtype=np.float32))

    if len(ctrls) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.stack(ctrls, axis=0)  # (N, 4)


def compute_counts(ctrls: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """
    ctrls : (N, 4)
    Retourne (counts_pos, total_frames)
    """
    if ctrls.size == 0:
        return np.zeros(4, dtype=np.int64), 0

    binary = ctrls > threshold
    counts_pos = binary.sum(axis=0)  # (4,)
    total = ctrls.shape[0]
    return counts_pos, total


def plot_distribution(counts_pos: np.ndarray,
                      total_frames: int,
                      title: str,
                      out_path: Path):
    """Trace un barplot des proportions d'activation par action."""
    if total_frames == 0:
        print(f"[WARN] Aucun frame pour {title}, pas de graphe.")
        return

    ratios = counts_pos / total_frames  # (4,)

    x = np.arange(len(ACTIONS))

    fig, ax = plt.subplots()
    bars = ax.bar(x, ratios)

    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion de frames actives")
    ax.set_title(title)

    # Annoter avec "count (xx%)"
    for i, b in enumerate(bars):
        h = b.get_height()
        txt = f"{counts_pos[i]} ({h*100:.1f}%)"
        ax.text(b.get_x() + b.get_width() / 2.0,
                h + 0.01,
                txt,
                ha="center",
                va="bottom",
                fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_stats(
    data_dir: str,
    pattern: str = "*.npz",
    threshold: float = 0.5,
    outdir: str = "stats_plots",
):
    """
    Calcule et trace la répartition des classes dans tous les fichiers du dossier.

    :param data_dir: dossier contenant les fichiers .npz (LZMA+pickle)
    :param pattern: pattern de fichiers, ex "*.npz"
    :param threshold: seuil pour considérer qu'une action est active
    :param outdir: dossier de sortie des graphes
    """
    data_dir = Path(data_dir)
    outdir = Path(outdir)

    files = sorted(data_dir.glob(pattern))
    if not files:
        print(f"Aucun fichier trouvé dans {data_dir} avec le pattern {pattern}")
        return

    global_counts = np.zeros(4, dtype=np.int64)
    global_total = 0

    print(f"Analyse de {len(files)} fichier(s) dans {data_dir}...")

    for f in files:
        print(f" - Lecture de {f.name} ...")
        try:
            ctrls = load_controls_from_file(f)
        except Exception as e:
            print(f"   [ERREUR] Impossible de lire {f}: {e}")
            continue

        counts_pos, total = compute_counts(ctrls, threshold)
        global_counts += counts_pos
        global_total += total

        print(f"   Frames: {total}")
        for i, act in enumerate(ACTIONS):
            ratio = (counts_pos[i] / total * 100) if total > 0 else 0.0
            print(f"   {act:7s}: {counts_pos[i]:5d} ({ratio:5.1f} %)")

        out_path = outdir / f"{f.stem}_distribution.png"
        plot_distribution(
            counts_pos,
            total,
            title=f"Répartition des classes - {f.name}",
            out_path=out_path,
        )

    # Graphe global
    print("\n=== Global (tous les fichiers) ===")
    if global_total > 0:
        for i, act in enumerate(ACTIONS):
            ratio = global_counts[i] / global_total * 100
            print(f"{act:7s}: {global_counts[i]:6d} ({ratio:5.1f} %)")

    global_out = outdir / "all_files_distribution.png"
    plot_distribution(
        global_counts,
        global_total,
        title="Répartition des classes - Tous les fichiers",
        out_path=global_out,
    )

    print(f"\nGraphes sauvegardés dans : {outdir.resolve()}")


if __name__ == "__main__":
    run_stats(
        data_dir="./",     # dossier où sont tes fichiers
        pattern="*.npz",          # motif de fichiers
        threshold=0.5,            # seuil d'activation
        outdir="stats_plots",     # dossier de sortie des graphs
    )
