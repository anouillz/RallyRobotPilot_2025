import argparse
import os
import textwrap
import numpy as np
import lzma
import pickle
from collections.abc import Mapping, Sequence


XZ_MAGIC = b"\xfd7zXZ\x00"


def summarize_array(arr, max_sample=6):
    try:
        arr = np.asarray(arr)
    except Exception:
        return str(type(arr))
    s = f"shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}, size={getattr(arr, 'size', None)}"
    if arr.size == 0:
        return s
    if np.issubdtype(arr.dtype, np.number):
        a = arr.ravel()
        return s + f", min={a.min():.6g}, max={a.max():.6g}, mean={a.mean():.6g}, std={a.std():.6g}, sample={a[:max_sample].tolist()}"
    else:
        flat = arr.ravel()
        n = min(len(flat), max_sample)
        return s + f", sample={flat[:n].tolist()}"


def inspect_numpy_npz(path, max_sample=6):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print("np.load failed:", e)
        return False

    print(textwrap.dedent(f"""
    Detected NumPy archive (.npz): {path}
    Keys: {list(data.keys())}
    """))

    for k in data.files:
        try:
            arr = data[k]
            print(f"- {k}: {summarize_array(arr, max_sample=max_sample)}")
        except Exception as e:
            print(f"- {k}: (error summarizing) {e}")
    return True


def inspect_lzma_pickle(path, max_items=5, max_sample=6):
    try:
        with lzma.open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print("Failed to lzma+pickle load:", e)
        return False

    print(textwrap.dedent(f"""
    Detected XZ-compressed pickle: {path}
    Top-level type: {type(obj)}
    """))

    def summarize_obj(o, prefix=""):
        t = type(o)
        if isinstance(o, np.ndarray):
            print(prefix + summarize_array(o, max_sample=max_sample))
        elif isinstance(o, Mapping):
            print(prefix + f"dict with keys: {list(o.keys())}")
            for kk in list(o.keys())[:max_items]:
                try:
                    v = o[kk]
                    print(prefix + f"  - {kk}: {type(v)} -> {summarize_array(v, max_sample=max_sample) if not isinstance(v, (Mapping, Sequence)) or isinstance(v, (str, bytes)) else type(v)}")
                except Exception as e:
                    print(prefix + f"  - {kk}: (error) {e}")
        elif isinstance(o, Sequence) and not isinstance(o, (str, bytes, bytearray)):
            n = len(o)
            print(prefix + f"sequence (len={n}), sample types: {[type(x) for x in list(o)[:max_items]]}")
            for i, x in enumerate(list(o)[:max_items]):
                print(prefix + f"  [{i}] type={type(x)}")
                if hasattr(x, '__dict__'):
                    d = getattr(x, '__dict__')
                    print(prefix + f"    attrs: {list(d.keys())}")
                    for kk, vv in list(d.items())[:max_items]:
                        try:
                            print(prefix + f"      - {kk}: {summarize_array(vv, max_sample=max_sample) if not isinstance(vv, (Mapping, Sequence)) or isinstance(vv, (str, bytes)) else type(vv)}")
                        except Exception as e:
                            print(prefix + f"      - {kk}: (error) {e}")
                else:
                    try:
                        print(prefix + f"    value summary: {summarize_array(x, max_sample=max_sample)}")
                    except Exception:
                        print(prefix + "    (no further summary)")
        else:
            # single object
            if hasattr(o, '__dict__'):
                d = getattr(o, '__dict__')
                print(prefix + f"object {t} with attrs: {list(d.keys())}")
                for kk, vv in list(d.items())[:max_items]:
                    try:
                        print(prefix + f"  - {kk}: {summarize_array(vv, max_sample=max_sample) if not isinstance(vv, (Mapping, Sequence)) or isinstance(vv, (str, bytes)) else type(vv)}")
                    except Exception as e:
                        print(prefix + f"  - {kk}: (error) {e}")
            else:
                print(prefix + f"value: {repr(o)[:200]}")

    summarize_obj(obj)
    return True


def main():
    p = argparse.ArgumentParser(description="Inspect a recording file (npz or xz-pickle)")
    p.add_argument('file', help='Path to recording file')
    p.add_argument('--max-items', type=int, default=5, help='Max items to sample from sequences or dicts')
    p.add_argument('--max-sample', type=int, default=6, help='Max numeric sample elements to print')
    args = p.parse_args()

    path = args.file
    if not os.path.exists(path):
        print('File not found:', path)
        return

    # Try numpy npz first
    ok = inspect_numpy_npz(path, max_sample=args.max_sample)
    if ok:
        return

    # lzma+pickle
    try:
        with open(path, 'rb') as fh:
            sig = fh.read(6)
    except Exception as e:
        print('Failed to read file header:', e)
        return

    if sig == XZ_MAGIC:
        ok2 = inspect_lzma_pickle(path, max_items=args.max_items, max_sample=args.max_sample)
        if ok2:
            return

    print('Unknown or unsupported file format')


if __name__ == '__main__':
    main()
