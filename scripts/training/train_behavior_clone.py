import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import RallyDataset
from models import CNNLSTMPolicy


def collate_fn(batch):
    imgs, labs = zip(*batch)
    if imgs[0].dim() == 4:
        imgs = torch.stack([i for i in imgs], dim=0)
    else:
        imgs = torch.stack(imgs, dim=0)
    labs = torch.stack(labs, dim=0)
    return imgs, labs


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])

    ds = RallyDataset(args.records_dir, transform=transform, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = CNNLSTMPolicy(use_lstm=args.use_lstm)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i, (imgs, labels) in enumerate(dl):
            imgs = imgs.to(device)
            labels = labels.to(device)
            if args.use_lstm and imgs.dim() == 5:
                # imgs: (B,T,C,H,W)
                preds = model(imgs)
            else:
                preds = model(imgs)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % args.log_step == 0:
                print(f'Epoch {epoch+1}/{args.epochs} Step {i+1}/{len(dl)} loss={loss.item():.4f}')

        t1 = time.time()
        print(f'Epoch {epoch+1} completed, avg loss={(epoch_loss/len(dl)):.4f}, time={(t1-t0):.1f}s')

        # save checkpoint
        ckpt = os.path.join(args.checkpoint_dir, f'bc_epoch{epoch+1}.pt')
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt)
        print('Saved checkpoint', ckpt)


def parse_args():
    p = argparse.ArgumentParser(description='Train behavior cloning CNN+LSTM')
    p.add_argument('--records-dir', default='records_c1', help='Directory containing recording .npz files')
    p.add_argument('--seq-len', type=int, default=1, help='Sequence length for LSTM (1 = per-frame)')
    p.add_argument('--use-lstm', action='store_true', help='Enable LSTM temporal head')
    p.add_argument('--width', type=int, default=160, help='Target image width')
    p.add_argument('--height', type=int, default=120, help='Target image height')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--checkpoint-dir', default='scripts/training/checkpoints')
    p.add_argument('--log-step', type=int, default=20)
    p.add_argument('--cpu', action='store_true', help='Force CPU')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
