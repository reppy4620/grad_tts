import torch

from pathlib import Path
from scipy.io import wavfile
from librosa.util import normalize
from dataclasses import dataclass

from text import phoneme_to_sequence, phonemes
from audio import mel_spectrogram


MAX_WAV_VALUE = 32767.


@dataclass
class Data:
    bname: str
    label: str


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, wav_dir, lab_dir, hop_size=240, sr=24000, split='|'):
        with open(file_path) as f:
            lines = f.readlines()
            data = list()
            for line in lines:
                bname, label = line.strip().split(split)
                data.append(Data(bname, label))
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.lab_dir = Path(lab_dir)
        
        self.hop_size = hop_size
        self.sr = sr

    def get_duration(self, filepath, text):
        with open(filepath, 'r') as f:
            labels = f.readlines()
        dur = []
        count = 0
        sum_quantized_diff_dur = 0
        extra_dur = 0
        for t in text.split():
            if t in phonemes or t in ['^', '$', '?', '_']:
                start, end, _ = labels[count].split(' ')
                start, end = int(start), int(end)
                raw_dur = (end - start) * 1e-7 / (self.hop_size / self.sr) + 0.001 # 浮動小数点数の丸め誤差の影響をなくす
                quantized_dur = int(raw_dur)
                sum_quantized_diff_dur += raw_dur - quantized_dur
                if sum_quantized_diff_dur >= 1.0:
                    dur.append(quantized_dur - extra_dur + 1)
                    sum_quantized_diff_dur = 0
                else:
                    dur.append(quantized_dur - extra_dur)
                count += 1
                extra_dur = 0
            else: # 特殊記号かつpauやsilでない場合
                dur.append(1)
                extra_dur = 1
        return torch.FloatTensor(dur), int(end * 1e-7 * self.sr + 0.001)


    def __getitem__(self, idx):
        d = self.data[idx]

        phonemes = torch.LongTensor(phoneme_to_sequence(d.label.split()))
        duration, audio_end_idx = self.get_duration(self.lab_dir / f'{d.bname}.lab', d.label)

        _, audio = wavfile.read(self.wav_dir / f'{d.bname}.wav')
        audio = audio / MAX_WAV_VALUE
        audio = audio[:audio_end_idx]
        audio = normalize(audio) * 0.95
        audio = torch.Tensor(audio).unsqueeze(0)
        mel = mel_spectrogram(audio).squeeze()

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
