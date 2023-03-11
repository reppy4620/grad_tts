import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels=192, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean)**2, dim=1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma[None, :, None] + self.beta[None, :, None]
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout=0., window_size=4):
        super().__init__()
        assert channels % n_heads == 0

        self.inter_channels = channels // n_heads
        self.n_heads = n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.inter_channels)

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

        rel_stddev = self.inter_channels ** -0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev)
        self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev)

    def forward(self, x, mask):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        B, C, T = q.size()
        query = q.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)
        key = k.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)
        value = v.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)

        scores = torch.matmul(query / self.scale, key.transpose(-2, -1))

        pad_length = T - (self.window_size + 1)
        if pad_length < 0:
            pad_length = 0
        start = (self.window_size + 1) - T
        if start < 0:
            start = 0
        end = start + 2 * T - 1

        pad_rel_emb = F.pad(self.emb_rel_k, [0, 0, pad_length, pad_length, 0, 0])
        k_emb = pad_rel_emb[:, start:end]

        rel_logits = torch.matmul(query / self.scale, k_emb.unsqueeze(0).transpose(-2, -1))
        rel_logits = F.pad(rel_logits, [0, 1])
        rel_logits = rel_logits.view([B, self.n_heads, 2 * T * T])
        rel_logits = F.pad(rel_logits, [0, T - 1])
        scores_local = rel_logits.view([B, self.n_heads, T + 1, 2 * T - 1])[:, :, :T, T - 1:]

        scores = scores + scores_local
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        p_attn = F.pad(p_attn, [0, T - 1])
        p_attn = p_attn.view([B, self.n_heads, T * (2 * T - 1)])
        p_attn = F.pad(p_attn, [T, 0])
        relative_weights = p_attn.view([B, self.n_heads, T, 2 * T])[:, :, :, 1:]

        pad_rel_emb = F.pad(self.emb_rel_v, [0, 0, pad_length, pad_length, 0, 0])
        v_emb = pad_rel_emb[:, start:end]

        output = output + torch.matmul(relative_weights, v_emb.unsqueeze(0))

        x = output.transpose(2, 3).contiguous().view(B, C, T)

        x = self.conv_o(x)
        return x



class FFN(nn.Module):
    def __init__(self, channels, scale=4, kernel_size=3, dropout=0.0):
        super(FFN, self).__init__()
        self.conv_1 = torch.nn.Conv1d(channels, channels*scale, kernel_size, padding=kernel_size//2)
        self.conv_2 = torch.nn.Conv1d(channels*scale, channels, kernel_size, padding=kernel_size//2)
        self.drop = torch.nn.Dropout(dropout)


    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
    


class AttentionLayer(nn.Module):
    def __init__(self, channels, num_head, dropout):
        super().__init__()
        self.attention_layer = MultiHeadAttention(channels, num_head, dropout)
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask):
        y = self.attention_layer(x, attn_mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x


class FFNLayer(nn.Module):
    def __init__(self, channels, scale=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.ffn = FFN(channels, scale, kernel_size, dropout)
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        y = self.ffn(x, mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x


class PreNetLayer(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm = LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.conv(x * mask)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x * mask


class PreNet(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.5, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNetLayer(channels, kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        residual = x
        for layer in self.layers:
            x = layer(x, mask)
        x = residual + x
        return x * mask
    

class EncoderLayer(nn.Module):
    def __init__(self, channels, num_head, kernel_size, dropout):
        super().__init__()
        self.attention = AttentionLayer(
            channels,
            num_head,
            dropout
        )
        self.ffn = FFNLayer(
            channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
    def forward(self, x, mask, attn_mask):
        x = self.attention(x, attn_mask)
        x = self.ffn(x, mask)
        return x


class Encoder(nn.Module):
    def __init__(self, num_vocab, channels, out_channels, num_head, num_layers, kernel_size, dropout):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, channels)
        self.scale = math.sqrt(channels)

        self.prenet = PreNet(channels)
        self.layers = nn.ModuleList([
            EncoderLayer(
                channels,
                num_head,
                kernel_size,
                dropout
            ) for _ in range(num_layers)
        ])
        self.postnet = nn.Conv1d(channels, out_channels, 1)

    def forward(self, x, mask):
        x = self.emb(x) * self.scale
        x = torch.transpose(x, 1, -1)
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        x = self.prenet(x, mask)
        for layer in self.layers:
            x = layer(x, mask, attn_mask)
        o = self.postnet(x) * mask
        return o, x
