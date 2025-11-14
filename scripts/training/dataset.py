import os
import lzma
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_recording(path: str):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)


class RallyDataset(Dataset):

    def __init__(self, records_dir: str, files: List[str] = None, transform=None, seq_len: int = 1):
        self.records_dir = records_dir
        if files is None:
            # find files with .npz extension
            files = [f for f in os.listdir(records_dir) if f.endswith('.npz')]
        self.files = [os.path.join(records_dir, f) for f in files]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.seq_len = max(1, int(seq_len))

        # Build an index mapping global idx -> (file_idx, local_idx)
        self.index = []
        for fi, fpath in enumerate(self.files):
            try:
                rec = load_recording(fpath)
                n = len(rec)
                # For sequence support
                for i in range(0, n - self.seq_len + 1):
                    self.index.append((fi, i))
            except Exception:
                # skip unreadable files
                continue

    def __len__(self):
        return len(self.index)

    def _get_snapshot(self, file_idx: int, local_idx: int):
        rec = load_recording(self.files[file_idx])
        return rec[local_idx]

    def __getitem__(self, idx: int):
        fi, li = self.index[idx]
        # Load sequence of snapshots
        imgs = []
        labels = []
        for t in range(self.seq_len):
            snap = self._get_snapshot(fi, li + t)
            img = getattr(snap, 'image')
            # image is HxWx3 uint8
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img)
            else:
                pil = Image.fromarray(np.asarray(img))

            if self.transform is not None:
                x = self.transform(pil)
            else:
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0

            imgs.append(x)

            cc = getattr(snap, 'current_controls')
            # Ensure tuple of 4 ints
            lab = torch.tensor([int(cc[0]), int(cc[1]), int(cc[2]), int(cc[3])], dtype=torch.float32)
            labels.append(lab)

        # Return sequence or single frame
        if self.seq_len == 1:
            return imgs[0], labels[0]
        else:
            imgs = torch.stack(imgs, dim=0)
            labels = torch.stack(labels, dim=0)
            return imgs, labels


if __name__ == '__main__':
    ds = RallyDataset('records_c1', seq_len=1)
    print('Found', len(ds), 'samples')
    x, y = ds[0]
    print('x', type(x), getattr(x, 'shape', None), 'y', y)
