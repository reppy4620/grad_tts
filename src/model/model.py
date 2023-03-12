import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .predictors import DurationPredictor
from .utils import sequence_mask, generate_path


class TTSModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.mel_dim = config.mel_dim
        self.seg_size = config.seg_size
        
        self.encoder = Encoder(**config.encoder)
        self.duration_predictor = DurationPredictor(**config.dp)
        self.decoder = Decoder(**config.decoder, mel_dim=self.mel_dim)

    @torch.no_grad()
    def forward(self, x, x_lengths):
        x_mask = sequence_mask(x_lengths, x.shape[-1]).unsqueeze(1).to(x)
        x, z = self.encoder(x, x_mask)
        duration = self.duration_predictor(z, x_mask)
        duration = torch.ceil(torch.exp(duration))
        y_lengths = torch.sum(duration, dim=(1, 2)).long()
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x)
        path_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        m = x @ attn_path
        y_lengths_max = (y_lengths.max() // self.decoder.scale) * self.decoder.scale
        m = m[..., :y_lengths_max]
        y_mask = y_mask[..., :y_lengths_max]
        x = m + torch.randn_like(m)
        o = self.decoder(x.unsqueeze(1), m.unsqueeze(1), y_mask.unsqueeze(1))
        return o, m

    def compute_loss(self, batch):
        (
            x,
            duration,
            x_lengths,
            y,
            y_lengths
        ) = batch
        x_mask = sequence_mask(x_lengths, x.shape[-1]).unsqueeze(1).to(x.dtype)
        x, z = self.encoder(x, x_mask)
        duration_pred = self.duration_predictor(z, x_mask)

        y_mask = sequence_mask(y_lengths, y.shape[-1]).unsqueeze(1).to(x)
        path_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        m = x @ attn_path

        off_set_ranges = list(zip([0] * len(y_lengths), (y_lengths - self.seg_size).clamp(0).cpu().numpy()))
        out_offset = torch.LongTensor([
            torch.tensor(random.choice(range(start, end)) if end > start else 0)
            for start, end in off_set_ranges
        ]).to(y_lengths)
        shape = (x.shape[0], x.shape[1], self.seg_size)
        m_cut = torch.zeros(*shape).to(m)
        y_cut = torch.zeros(*shape).to(y)
        y_mask_cut = torch.zeros(x.shape[0], 1, self.seg_size).to(y_mask)
        for i, offset in enumerate(out_offset):
            cut_length = self.seg_size + (y_lengths[i] - self.seg_size).clamp(None, 0)
            l, u = offset, offset + cut_length
            m_cut[i, :, :cut_length] = m[i, :, l:u]
            y_cut[i, :, :cut_length] = y[i, :, l:u]
            y_mask_cut[i, :, :cut_length] = y_mask[i, :, l:u]

        diffusion_loss = self.decoder.compute_loss(y_cut.unsqueeze(1), m_cut.unsqueeze(1), y_mask_cut.unsqueeze(1))
        # eliminate constant term of gaussian negative log likelihood , so rest of equation will be mse_loss
        mel_loss = (0.5 * (y - m) ** 2).sum() / self.mel_dim / y_lengths.sum()
        log_duration = torch.log(1e-5 + duration) * x_mask
        duration_loss = F.mse_loss(duration_pred, log_duration, reduction='sum') / x_lengths.sum()
        loss = mel_loss + duration_loss + diffusion_loss
        return dict(
            loss=loss,
            mel_mean=mel_loss,
            duration=duration_loss,
            diffusion=diffusion_loss,
        )
