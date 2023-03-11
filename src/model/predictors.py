import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LayerNorm


class Block(nn.Module):
    def __init__(self, channels, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm = LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x * mask


class Layer(nn.Module):
    def __init__(self, channels, kernel_size, dropout) -> None:
        super().__init__()
        self.layer1 = Block(channels, kernel_size)
        self.layer2 = Block(channels, kernel_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        y = self.layer1(x, mask)
        y = self.dropout(y)
        y = self.layer2(y, mask)
        return x + y


class DurationPredictor(nn.Module):
    def __init__(self, channels, kernel_size, dropout, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(channels, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.out_layer = nn.Conv1d(channels, 1, 1)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.out_layer(x) * mask
        return x