import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import config as cfg
from model import TTSModel
from dataset import TextMelDataset, collate_fn
from utils import Tracker, seed_everything


def main(args):
    seed_everything()

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TTSModel(cfg.model).to(device)

    train_ds = TextMelDataset(args.train_file, args.wav_dir, args.lab_dir)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    valid_ds = TextMelDataset(args.valid_file, args.wav_dir, args.lab_dir)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)

    start_epoch = 1
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Loaded from checkpoint')
        print(f'Start from Epoch {start_epoch}')

    def to_device(*args):
        return (a.to(device) for a in args)

    def handle_batch(batch):
        _, *data = batch
        data = to_device(*data)
        loss = model.compute_loss(data)
        return loss

    tracker = Tracker(output_dir / f'loss.csv', mode='a' if args.ckpt_path is not None else 'w')
    for epoch in range(start_epoch, cfg.train.num_epoch+1):
        bar = tqdm(train_dl, total=len(train_dl), desc=f'Epoch: {epoch}')
        model.train()
        for batch in bar:
            optimizer.zero_grad()
            loss_dict = handle_batch(batch)
            loss = loss_dict['loss']
            loss.backward()
            _ = clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tracker.update(**{f'train_{k}': v.item() for k, v in loss_dict.items()})
            s = ', '.join(f'{k.split("_")[1]}: {v.mean():.5f}' for k, v in tracker.items())
            bar.set_postfix_str(s)
        
        model.eval()
        for batch in valid_dl:
            with torch.no_grad():
                loss_dict = handle_batch(batch)
            tracker.update(**{f'valid_{k}': v.item() for k, v in loss_dict.items()})

        save_obj = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(save_obj, ckpt_dir / 'last.pth')
        if epoch % cfg.train.save_interval == 0:
            torch.save(save_obj, ckpt_dir / f'epoch-{epoch}.pth')
        tracker.write(epoch, clear=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--lab_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    main(args)
