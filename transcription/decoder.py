import torch as th
from torch import nn
from typing import List

class NeighborhoodAttention2D_diffusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, cond=None, t=None):
        return x, cond, t

class NeighborhoodCrossAttention2D_diffusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, cond=None, t=None):
        return x, cond, t

class NeighborhoodAttention1D_diffusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, t=None):
        return x, t

class Decoder(nn.Module):
    def __init__(self,
                label_embed_dim: int,
                lstm_dim: int,
                n_layers: int,
                window: List[int],
                dilation: List[int],
                condition_method: str,
                diffusion_step: int,
                timestep_type: str,
                natten_direction: str,
                spatial_size: List[int],
                num_state: int = 5,
                classifier_free_guidance: bool = False,
                 ):
        super().__init__()
        self.label_embed_dim = label_embed_dim
        features_embed_dim = 128
        self.n_unit = lstm_dim
        self.n_layers = n_layers
        self.window = window
        self.cross_condition = condition_method
        self.diffusion_step = diffusion_step
        self.timestep_type = timestep_type
        self.natten_direction = natten_direction
        self.spatial_size = spatial_size
        self.dilation = dilation
        self.num_state = num_state
        self.classifier_free_guidance = classifier_free_guidance

        self.trans_model = LSTM_NATTEN((label_embed_dim + features_embed_dim),
                                        timestep_type=self.timestep_type,
                                        diffusion_step=self.diffusion_step,
                                        natten_direction=self.natten_direction,
                                        spatial_size=self.spatial_size,
                                        dilation=self.dilation,
                                        window=self.window,
                                        n_unit=self.n_unit,
                                        n_layers=self.n_layers,
                                        cross_condition=self.cross_condition)
        self.output = nn.Linear(self.n_unit, 5)
        self.label_emb = nn.Embedding(num_state + 1, label_embed_dim)

        if classifier_free_guidance:
            self.use_cfg = True
            self.null_features_emb = nn.Parameter(th.randn(1, 128))
        else:
            self.use_cfg = False

    def forward(self, label, feature, t, cond_drop_prob=None):
        batch = label.shape[0]

        if self.use_cfg:
            if cond_drop_prob is None:
                cond_drop_prob = 0.0

            keep_mask = th.zeros((batch, 1, 1), device=label.device).float().uniform_(0, 1) < (1 - cond_drop_prob)
            null_cond_emb = self.null_features_emb.repeat(label.shape[0], feature.shape[1], 1)
            feature = th.where(keep_mask, feature, null_cond_emb)

        label_emb = self.label_emb(label)

        if label_emb.ndim == 4:
            label_emb = label_emb.permute(0, 2, 3, 1).reshape(label_emb.shape[0], -1, label_emb.shape[1])

        if label_emb.shape[1] != feature.shape[1]:
            raise ValueError(f"Mismatch in time steps between label_emb and feature: {label_emb.shape[1]} vs {feature.shape[1]}")

        input_feature = th.cat((label_emb, feature), dim=-1)

        if self.cross_condition == 'self':
            x = self.trans_model(input_feature, None, t)
        elif self.cross_condition in ['cross', 'self_cross']:
            x = self.trans_model(input_feature, feature, t)
        else:
            x = self.trans_model(input_feature, None, t)

        out = self.output(x)
        return out.reshape(batch, -1, self.output.out_features)

class LSTM_NATTEN(nn.Module):
    def __init__(self,
                 hidden,
                 timestep_type,
                 diffusion_step,
                 natten_direction,
                 spatial_size,
                 dilation: List[int],
                 window=25,
                 n_unit=24,
                 n_head=4,
                 n_layers=2,
                 cross_condition=False):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_unit = n_unit
        self.timestep_type = timestep_type
        self.diffusion_step = diffusion_step
        self.lstm = nn.LSTM(hidden, n_unit // 2, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.natten_dir = natten_direction
        self.spatial_size = spatial_size
        self.cross_condition = cross_condition
        self.dilation = dilation
        self.na = nn.ModuleList([NeighborhoodAttention2D_diffusion() for _ in range(n_layers)])

    def forward(self, x, cond=None, t=None):
        BATCH, SEQ_LEN, HIDDEN = x.shape
        x, _ = self.lstm(x)
        assert x.shape[1] == 88, f"Expected LSTM output sequence length T=88, but got T={x.shape[1]}"
        x = x.unsqueeze(2)
        for layer in self.na:
            x, _, _ = layer(x, cond, t)
        return x