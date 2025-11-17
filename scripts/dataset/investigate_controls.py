import numpy as np
from collections import Counter

TRAIN_PATH = "scripts/dataset/all_train_seq5.npz"
VAL_PATH   = "scripts/dataset/all_val_seq5.npz" 


def load_Y(path):
    print(f"Loading {path} ...")
    d = np.load(path)
    return d["Y"] 


def main():
    #labels 
    Y_train = load_Y(TRAIN_PATH)
    try:
        Y_val = load_Y(VAL_PATH)
        Y = np.concatenate([Y_train, Y_val], axis=0)
    except FileNotFoundError:
        print("Validation file not found, using only train.")
        Y = Y_train

    print("\nTotal samples:", len(Y))

    W = Y[:, 0]
    S = Y[:, 1]
    A = Y[:, 2]
    D = Y[:, 3]

    def stats(name, arr):
        pressed = (arr > 0.5).sum()
        print(f"{name}: pressed in {pressed} samples ({pressed / len(arr) * 100:.2f}%)")

    print("\nPer-key usage:")
    stats("W (forward)", W)
    stats("S (back)", S)
    stats("A (left)", A)
    stats("D (right)", D)

    # move vs no-move
    idle_mask = (Y.sum(axis=1) == 0)
    num_idle = idle_mask.sum()
    num_non_idle = len(Y) - num_idle

    print("\nIdle vs non-idle:")
    print(f"Idle : {num_idle} samples ({num_idle / len(Y) * 100:.2f}%)")
    print(f"Non-idle : {num_non_idle} samples ({num_non_idle / len(Y) * 100:.2f}%)")

    # combination counts
    combos = [tuple(row.tolist()) for row in Y]
    counter = Counter(combos)

    print("\nTop 10 most common control combinations (W, S, A, D):")
    for combo, count in counter.most_common(10):
        print(f"{combo}: {count} samples ({count / len(Y) * 100:.2f}%)")


if __name__ == "__main__":
    main()
