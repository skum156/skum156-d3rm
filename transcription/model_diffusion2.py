import torch as th
from torch import nn
import numpy as np
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms

# from .cqt import CQT
from .constants import SR, HOP
from .context import random_modification, update_context
# from .cqt import MultiCQT
from .midispectrogram import CombinedSpec, MidiSpec
from .model import MIDIFrontEnd, FilmLayer, HarmonicDilatedConv
from .diffusion.natten_diffusion import NeighborhoodAttention2D_diffusion, NeighborhoodAttention1D_diffusion
from .diffusion.embedding.dalle_mask_image_embedding import DalleMaskImageEmbedding
from .pretrain import load_pretrain

class TransModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        # self.hidden_per_pitch = config.hidden_per_pitch
        self.label_embed_dim = config.model_config["label_emb_config"]["label_embed_dim"]
        self.features_embed_dim = 128 # from pretrained model
        self.n_unit = config.model_config["lstm_natten_config"]["n_unit"]
        self.n_layers = config.model_config["lstm_natten_config"]["n_layers"]
        self.window= config.model_config["lstm_natten_config"]["window"]
        self.pitchwise = config.pitchwise_lstm
        self.devices = device

        # self.trans_model = NATTEN(config.hidden_per_pitch)
        self.trans_model = LSTM_NATTEN((self.label_embed_dim+self.features_embed_dim), config,
                                       window=self.window,
                                       n_unit=self.n_unit,
                                       n_layers=self.n_layers
                                       )
        self.output = nn.Linear(self.n_unit, 5)
        # self.embedding = nn.Embedding(5, 4)
        self.label_emb = DalleMaskImageEmbedding(num_embed=config.model_config["label_emb_config"]["num_embed"],
                                                   spatial_size=config.model_config["label_emb_config"]["spatial_size"],
                                                   embed_dim=config.model_config["label_emb_config"]["label_embed_dim"],
                                                   trainable=config.model_config["label_emb_config"]["trainable"],
                                                   pos_emb_type=config.model_config["label_emb_config"]["pos_emb_type"])
        # Load pretrained model
        self.pretrain_model = load_pretrain(use_vel=True)

    def forward(self, label, audio, t):
        # features (=cond_emb) : B x T*88 x H
        # label (=x_t) : B x T*88 x 1
        # features are concatanated with label as input to model
        features = self.pretrain_model(audio, device=self.devices)
        features = features.transpose(-2, -1) # B x T x 88 x H(=128)
        features = features.reshape(features.shape[0], -1, features.shape[-1]) # B x T*88 x H
        label_emb = self.label_emb(label) # B x T*88 x label_embed_dim
        features = th.cat((label_emb, features), dim=-1) # TODO - check if features shape = B x T*88 x (H + label_embed_dim)
        x = self.trans_model(features, t)
        out = self.output(x)
        return out.reshape(x.shape[0], x.shape[1]*x.shape[2], -1).permute(0, 2, 1) # B x 5 x T*88

def get_conv2d_block(channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1],
                        use_film=False, n_f=None):
    modules = [
        nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
        nn.ReLU()
        ]
    if use_film:
        modules.append(FilmLayer(n_f, channel_out))
    if pool_size != None:
        modules.append(nn.MaxPool2d(pool_size))
    modules.append(nn.InstanceNorm2d(channel_out))
    return nn.Sequential(*modules)


class HPP_FC(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, hidden_per_pitch, use_film,
                 n_per_pitch=5, frontend_kernel_size=7):
        super().__init__()
        # input is batch_size * 3 channel * frames * n_mels
        
        self.n_h = hidden_per_pitch // 2
        self.conv_front = nn.Sequential(
            get_conv2d_block(3, cnn_unit, kernel_size=frontend_kernel_size, use_film=use_film, n_f=n_mels),
            get_conv2d_block(cnn_unit, cnn_unit, kernel_size=frontend_kernel_size, use_film=use_film, n_f=n_mels),
            get_conv2d_block(cnn_unit, cnn_unit, kernel_size=frontend_kernel_size, use_film=use_film, n_f=n_mels))
        self.hdc0 = nn.Sequential(
            HarmonicDilatedConv(cnn_unit, self.n_h, n_per_pitch),
             HarmonicDilatedConv(self.n_h, self.n_h, n_per_pitch),
             HarmonicDilatedConv(self.n_h, self.n_h, n_per_pitch),
            get_conv2d_block(self.n_h, self.n_h, pool_size=[1, n_per_pitch], 
                                  dilation=[1, 12*n_per_pitch]),
            )
        self.hdc1 = nn.Sequential(
            get_conv2d_block(self.n_h, self.n_h, dilation=[1, 12]),
            get_conv2d_block(self.n_h, self.n_h, [5,1]),
            get_conv2d_block(self.n_h, self.n_h, [5,1]),
            get_conv2d_block(self.n_h, self.n_h, [5,1])
            )
        
        self.conv = get_conv2d_block(self.n_h, 8, [1,1])
        self.fc = nn.Sequential(
            nn.Linear(8*99, fc_unit),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(fc_unit, self.n_h*88),
            nn.LayerNorm(self.n_h*88),
            nn.ReLU()
            )
        
    def forward(self, x):
        # x: B x 3 x n_mels x T 
        batch_size = x.shape[0]
        T = x.shape[3]
        x = self.conv_front(x.permute(0, 1, 3, 2))
        x = self.hdc0(x)  # B, 128, T, 99
        x_hdc = self.hdc1(x[:,:,:,:88])  # B, 128, T, 88
        x_conv = self.conv(x)  # B, 8, T, 99
        x_fc = self.fc(x_conv.permute(0,2,1,3).flatten(-2)).reshape(batch_size, T, self.n_h, 88)
        x_fc = x_fc.permute(0, 2, 1, 3) # B, H/2, T, 88
        x = th.cat((x_hdc, x_fc), dim=1)  # B, H, T, 88
        return x.permute(0, 2, 3, 1)  # B, T, 88, H
    

class NATTEN(nn.Module):
    def __init__(self, hidden_per_pitch, window=25, n_unit=24, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers

        self.linear = nn.Sequential(nn.Linear(hidden_per_pitch+5, n_unit),
                                    nn.ReLU())
        self.na = nn.Sequential(*([NeighborhoodAttention2D_diffusion(n_unit, 4, window)]* n_layers))


    def forward(self, x):
        # x: B x T x 88 x H+5
        cat = self.linear(x)
        na_out = self.na(cat) # B x T x 88 x N
        return na_out
        
class LSTM_NATTEN(nn.Module):
    def __init__(self, hidden, config, window=25, n_unit=24, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_unit = n_unit
        self.timestep_type= config.model_config['params']['timestep_type']
        self.diffusion_step= config.diffusion_config['params']['diffusion_step']
        self.lstm = nn.LSTM(hidden, n_unit//2, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.natten_dir = config.model_config['params']['natten_direction']
        if self.natten_dir == '1d':
            self.na = nn.ModuleList([(NeighborhoodAttention1D_diffusion(n_unit, 1, window[0],
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                    NeighborhoodAttention1D_diffusion(n_unit, 1, window[1],
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                     
                                    )] * n_layers)
        elif self.natten_dir == '2d':
            self.na = nn.ModuleList([NeighborhoodAttention2D_diffusion(n_unit, 4, window,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type)] * n_layers)

    def forward(self, x, t):
        """
        x shape : B x T*88 x H
        """
        B = x.shape[0]
        H = x.shape[-1]
        T = x.shape[1]//88
        x = x.reshape(B, T, 88, H) # B x T x 88 x H
        x = x.permute(0, 2, 1, 3).reshape(B*88, T, H) # B*88 x T x H
        x, _ = self.lstm(x) 
        if self.natten_dir == '1d':
            for layer_1t, layer_1f in self.na:
                x, t = layer_1t(x, t)
                x = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3).reshape(B*T, 88, self.n_unit)
                x, t = layer_1f(x, t)
                x = x.reshape(B, T, 88, self.n_unit).permute(0,2,1,3).reshape(B*88, T, self.n_unit)
            x = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3) # B x T x 88 x
        
        elif self.natten_dir == '2d':
            x = x.reshape(B, 88, T, -1).permute(0,2,1,3) # B x T x 88 x H
            for layers in self.na:
                x, t = layers(x, t)

        return x
