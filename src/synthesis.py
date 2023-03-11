import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import config as cfg
from model import TTSModel
from dataset import TextMelDataset, collate_fn
from utils import seed_everything


def main(args):
    seed_everything()

    output_dir = Path(args.output_dir)
    mel_dir = output_dir / 'mel'
    mean_dir = output_dir / 'mean'
    plot_dir = output_dir / 'plot'
    [d.mkdir(parents=True, exist_ok=True) for d in [mel_dir, mean_dir, plot_dir]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TTSModel(cfg.model).to(device).eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['model'])

    ds = TextMelDataset(args.text_file, args.wav_dir, args.lab_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    def to_device(*args):
        return (a.to(device) for a in args)

    bar = tqdm(dl, total=len(dl))
    for batch in bar:
        (
            bnames,
            x,
            _,
            x_lengths,
            *_
        ) = batch
        bname = bnames[0]
        bar.set_description_str(bname)
        x, x_lengths = to_device(x, x_lengths)
        with torch.no_grad():
            mel, mean = model(x, x_lengths)
        mel = mel.squeeze().cpu().numpy()
        mean = mean.squeeze().cpu().numpy()
        np.save(mel_dir / f'{bname}.npy', mel)
        np.save(mean_dir / f'{bname}.npy', mean)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(mel, origin='lower', aspect='auto')
        ax[1].imshow(mean, origin='lower', aspect='auto')
        plt.savefig(plot_dir / f'{bname}.png')
        plt.close()
        break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text_file', type=str, required=True)
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--lab_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    main(args)
