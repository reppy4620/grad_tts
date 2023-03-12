import numpy as np
import torch

from pathlib import Path
from scipy.io import wavfile
from librosa.util import normalize
from dataclasses import dataclass

from text import text_to_sequence, phonemes
from audio import mel_spectrogram


MAX_WAV_VALUE = 32767.


@dataclass
class Data:
    bname: str
    label: str


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, wav_dir, lab_dir, mel_config, split='|'):
        with open(file_path) as f:
            lines = f.readlines()
            data = list()
            for line in lines:
                bname, label = line.strip().split(split)
                data.append(Data(bname, label))
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.lab_dir = Path(lab_dir)
        
        self.mel_config = mel_config
        self.hop_size = mel_config.hop_size
        self.sampling_rate = mel_config.sampling_rate

    def get_duration(self, filepath, label, mel_length):
        with open(filepath, 'r') as f:
            labels = f.readlines()
        durations = []
        cnt = 0
        for s in label.split():
            if s in phonemes or s in ['^', '$', '?', '_']:
                s, e, _ = labels[cnt].split()
                s, e = int(s), int(e)
                dur = (e - s) * 1e-7 / (self.hop_size / self.sampling_rate)
                durations.append(dur)
                cnt += 1
            else:
                durations.append(1)
        # adjust length, differences caused by round op.
        round_durations = np.round(durations)
        diff_length = np.sum(round_durations) - mel_length
        if diff_length == 0:
            return torch.FloatTensor(round_durations)
        elif diff_length > 0:
            durations_diff = round_durations - durations
            d = -1
        else: # diff_length < 0
            durations_diff = durations - round_durations
            d = 1
        sort_dur_idx = np.argsort(durations_diff)[::-1]
        for i, idx in enumerate(sort_dur_idx, start=1):
            round_durations[idx] += d
            if i == abs(diff_length):
                break
        assert np.sum(round_durations) == mel_length
        return torch.FloatTensor(round_durations)


    def __getitem__(self, idx):
        d = self.data[idx]

        phonemes = torch.LongTensor(text_to_sequence(d.label.split()))

        _, wav = wavfile.read(self.wav_dir / f'{d.bname}.wav')
        wav = wav / MAX_WAV_VALUE
        wav = normalize(wav) * 0.95
        wav = torch.Tensor(wav).unsqueeze(0)
        mel = mel_spectrogram(wav, **self.mel_config).squeeze()
        mel_length = mel.size(-1)

        duration = self.get_duration(self.lab_dir / f'{d.bname}.lab', d.label, mel_length)

        return (
            d.bname,
            phonemes,
            duration,
            mel
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    (
        bnames,
        phonemes,
        durations,
        mels
    ) = tuple(zip(*batch))

    B = len(bnames)
    x_lengths = [len(x) for x in phonemes]
    y_lengths = [x.size(1) for x in mels]

    x_max_length = max(x_lengths)
    y_max_length = max(y_lengths)
    mel_dim = mels[0].size(0)

    x_pad = torch.zeros(size=(B, x_max_length), dtype=torch.long)
    d_pad = torch.zeros(size=(B, 1, x_max_length), dtype=torch.float)
    y_pad = torch.zeros(size=(B, mel_dim, y_max_length), dtype=torch.float)
    for i, (p, d, m) in enumerate(zip(phonemes, durations, mels)):
        x_l, d_l, m_l = x_lengths[i], x_lengths[i], y_lengths[i]
        x_pad[i, :x_l] = p
        d_pad[i, :, :d_l] = d
        y_pad[i, :, :m_l] = m

    x_lengths = torch.LongTensor(x_lengths)
    y_lengths = torch.LongTensor(y_lengths)

    return (
        bnames,
        x_pad,
        d_pad,
        x_lengths,
        y_pad,
        y_lengths,
    )
