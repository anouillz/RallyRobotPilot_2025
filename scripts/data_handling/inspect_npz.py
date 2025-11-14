#keys, shapes, dtypes and a small numeric summary/sample for each array.

import argparse
import numpy as np
import os
import textwrap


def summarize_array(name, arr, max_sample=6):
    print(f"- {name}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}, size={getattr(arr, 'size', None)}")
    try:
        if arr.size == 0:
            print("  (empty array)")
            return
    except Exception:
        pass

    # Numeric summaries
    try:
        if np.issubdtype(getattr(arr, 'dtype', object), np.number):
            a = arr.astype(np.float64).ravel()
            print(f"  min={a.min():.6g}, max={a.max():.6g}, mean={a.mean():.6g}, std={a.std():.6g}")
            # show a few values
            print("  sample:", a[:max_sample].tolist())
            return
    except Exception:
        pass

    # For object / string arrays, show small sample
    try:
        flat = np.asarray(arr).ravel()
        n = min(len(flat), max_sample)
        print("  sample:", flat[:n].tolist())
    except Exception as e:
        print("  (could not sample values)", e)


def main():
    p = argparse.ArgumentParser(description="Inspect a .npz recording file and print contained arrays and summaries")
    p.add_argument("npzfile", help="Path to .npz file")
    p.add_argument("--max-sample", type=int, default=6, help="Max items to print per array")
    args = p.parse_args()

    path = args.npzfile
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print("Failed to load .npz:", e)
        return

    print(textwrap.dedent(f"""
    File: {path}
    Keys: {list(data.keys())}
    """))

    for k in data.files:
        try:
            arr = data[k]
            summarize_array(k, arr, max_sample=args.max_sample)
        except Exception as e:
            print(f"- {k}: (error reading) {e}")


if __name__ == "__main__":
    main()
