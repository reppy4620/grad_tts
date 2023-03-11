from text import num_symbol

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


train = DotDict({
    'num_epoch': 1000,
    'batch_size': 32,
    'save_interval': 100
})

channels = 192
mel_dim = 80
dim = 64

model = DotDict({
    'mel_dim': mel_dim,
    'out_size': 256
})
model.encoder = DotDict({
    'num_vocab': num_symbol(),
    'channels': channels,
    'out_channels': mel_dim,
    'num_head': 2,
    'num_layers': 6,
    'kernel_size': 3,
    'dropout': 0.1,
})
model.dp = DotDict({
    'channels': channels,
    'kernel_size': 3,
    'dropout': 0.1,
    'num_layers': 2
})
model.decoder = DotDict({
    'dim': dim,
    'channels': 1,
    'dim_mults': (1, 2, 4),
})

optimizer = DotDict({
    'lr': 1e-4
})
