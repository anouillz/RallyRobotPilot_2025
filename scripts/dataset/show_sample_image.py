import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "scripts/dataset/c1_train_seq5.npz"

CONTROL_ORDER = ["W (forward)", "S (back)", "A (left)", "D (right)"]


def main():
    data = np.load(DATASET_PATH)
    X = data["X"]   # shape: (N, seq_len, 120, 160, 3)
    Y = data["Y"]   # shape: (N, 4)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # idx of sequence and frame to show
    seq_idx = 0
    frame_idx = 0   

    img = X[seq_idx, frame_idx]   
    controls = Y[seq_idx]         

    print(f"\nSequence index: {seq_idx}")
    print(f"Showing frame index: {frame_idx} of that sequence")
    print("Raw controls (W, S, A, D):", controls.tolist())

    # keys pressed
    pressed = [
        name for value, name in zip(controls, CONTROL_ORDER)
        if value > 0.5
    ]
    if not pressed:
        pressed_str = "No key pressed"
    else:
        pressed_str = ", ".join(pressed)

    print("Interpreted controls:", pressed_str)

    plt.imshow(img.astype(np.uint8))
    plt.axis("off")
    plt.title(f"Seq {seq_idx}, frame {frame_idx}\nControls (last frame): {pressed_str}")
    plt.show()


if __name__ == "__main__":
    main()
