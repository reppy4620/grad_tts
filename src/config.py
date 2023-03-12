from text import num_vocab

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


train = DotDict(
    num_epoch=1000,
    batch_size=32,
    save_interval=100
)

mel_dim = 80

mel = DotDict(
    n_fft=1024, 
    num_mels=mel_dim, 
    sampling_rate=24000, 
    hop_size=240, 
    win_size=1024, 
    fmin=0, 
    fmax=12000, 
    center=False
)

channels = 192

model = DotDict(
    mel_dim=mel_dim,
    seg_size=256,
)
model.encoder = DotDict(
    num_vocab=num_vocab(),
    channels=channels,
    out_channels=mel_dim,
    num_head=2,
    num_layers=6,
    kernel_size=3,
    dropout=0.1,
    window_size=4
)
model.dp = DotDict(
    channels=channels,
    kernel_size=3,
    dropout=0.1,
    num_layers=3
)
model.decoder = DotDict(
    dim=64,
    channels=1,
    dim_mults=(1, 2, 4)
)

optimizer = DotDict(
    lr=1e-4
)
