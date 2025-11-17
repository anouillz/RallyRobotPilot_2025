import os
import lzma
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


RECORDS_DIR = "records_all"     
SEQUENCE_LENGTH = 10

IMG_W = 160
IMG_H = 120

TRAIN_OUTPUT = "all_train_seq5.npz"
VAL_OUTPUT   = "all_val_seq5.npz"


def load_record(path):
    with lzma.open(path, "rb") as f:
        snapshots = pickle.load(f)
    return snapshots


# preprocess images
def preprocess_image(img):
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((IMG_W, IMG_H))
    return np.array(pil_img, dtype=np.uint8)



def build_full_sequence_dataset():
    frames = []
    controls = []

    files = [
        os.path.join(RECORDS_DIR, f)
        for f in os.listdir(RECORDS_DIR)
        if f.endswith(".npz") or f.endswith(".lzma")
    ]

    print(f"Found {len(files)} record files.\n")

    for fpath in tqdm(files, desc="Loading records"):
        snapshots = load_record(fpath)

        for snap in snapshots:
            if snap.image is None or snap.current_controls is None:
                continue

            frames.append(preprocess_image(snap.image))
            controls.append(np.array(snap.current_controls, dtype=np.int8))

    print(f"\nTotal frames collected: {len(frames)}")

    # sliding window
    X, Y = [], []

    print("\nBuilding sequences...")
    for i in range(len(frames) - SEQUENCE_LENGTH):
        seq = frames[i:i + SEQUENCE_LENGTH]
        target = controls[i + SEQUENCE_LENGTH - 1]

        X.append(seq)
        Y.append(target)

    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y, dtype=np.int8)

    print("Final shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)

    return X, Y


# split into train and validation sets
def split_dataset(X, Y, val_ratio=0.2):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=val_ratio, shuffle=True, random_state=42
    )
    return X_train, X_val, Y_train, Y_val



def save_dataset(X_train, X_val, Y_train, Y_val):
    np.savez_compressed(TRAIN_OUTPUT, X=X_train, Y=Y_train)
    np.savez_compressed(VAL_OUTPUT,   X=X_val,   Y=Y_val)
    print(f"\nSaved:")
    print(f" - {TRAIN_OUTPUT}")
    print(f" - {VAL_OUTPUT}")


if __name__ == "__main__":
    X, Y = build_full_sequence_dataset()
    X_train, X_val, Y_train, Y_val = split_dataset(X, Y)
    save_dataset(X_train, X_val, Y_train, Y_val)
