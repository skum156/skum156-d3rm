import torch as th
from torch import nn
import numpy as np
from torch.nn import functional as F
import nnAudio.Spectrogram
from torchaudio import transforms

# from .cqt import CQT
from .constants import SR, HOP
from .context import random_modification, update_context
# from .cqt import MultiCQT
from .midispectrogram import CombinedSpec, MidiSpec
from .model import MIDIFrontEnd, FilmLayer, HarmonicDilatedConv
from .diffusion.natten_diffusion import NeighborhoodAttention2D_diffusion, NeighborhoodAttention1D_diffusion, NeighborhoodCrossAttention2D_diffusion, NeighborhoodAttention2D_diffusion_encoder
# from .diffusion.embedding.dalle_mask_image_embedding import DalleMaskImageEmbedding
from .pretrain import load_pretrain



class TransModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        # self.hidden_per_pitch = config.hidden_per_pitch
        self.label_embed_dim = config.model_config["label_emb_config"]["label_embed_dim"]
        if config.local_model_name == 'NAT_encoder': self.features_embed_dim = config.model_config["natten_encoder_config"]["n_unit"]
        else: self.features_embed_dim = 128 # from pretrained model
        if config.use_vel: self.features_embed_dim = self.features_embed_dim*2 # from pretrained model
        self.trained_encoder = config.model_config["params"]["trained_encoder"]
        self.encoder_type = config.model_config["params"]["encoder_type"]
        self.n_unit = config.model_config["lstm_natten_config"]["n_unit"]
        self.n_layers = config.model_config["lstm_natten_config"]["n_layers"]
        self.window = config.model_config["lstm_natten_config"]["window"]
        self.pitchwise = config.pitchwise_lstm
        self.cross_condition = config.model_config["params"]["cross_condition"]
        self.diffusion_step = config.diffusion_config["params"]["diffusion_step"]
        self.devices = device

        # self.trans_model = NATTEN(config.hidden_per_pitch)
        self.trans_model = LSTM_NATTEN((self.label_embed_dim+self.features_embed_dim), config,
                                       window=self.window,
                                       n_unit=self.n_unit,
                                       n_layers=self.n_layers,
                                       cross_condition=self.cross_condition,
                                       )
        self.output = nn.Linear(self.n_unit, 5)
        self.label_emb = nn.Embedding(config.model_config["label_emb_config"]["num_embed"] + 1, # +1 is for mask
                                      config.model_config["label_emb_config"]["label_embed_dim"])

        # classifier_free_guidance
        if config.diffusion_config["params"]["classifier_free_guidance"]:
            self.use_cfg = True
            self.cond_scale = config.diffusion_config["params"]["classifier_free_guidance"]["cond_scale"]
            self.cond_drop_prob = config.diffusion_config["params"]["classifier_free_guidance"]["cond_drop_prob"]
            self.null_features_emb = nn.Parameter(th.randn(1, 128)) # null embedding for cfg
        else:
            self.use_cfg = False
        

        # self.label_emb = DalleMaskImageEmbedding(num_embed=config.model_config["label_emb_config"]["num_embed"],
        #                                            spatial_size=config.model_config["label_emb_config"]["spatial_size"],
        #                                            embed_dim=config.model_config["label_emb_config"]["label_embed_dim"],
        #                                            trainable=config.model_config["label_emb_config"]["trainable"],
        #                                            pos_emb_type=config.model_config["label_emb_config"]["pos_emb_type"])

        # Load pretrained model
        if self.encoder_type == "CNN": 
            if not self.trained_encoder:
                # print('Loading & Training CNN Model from scratch')
                # self.pretrain_model = CNNModel(config, use_vel=config.use_vel)
                print('Loading & Training HPP Net from scratch')
                self.pretrain_model = HPPNet(config)
            elif self.trained_encoder:
                print('Loading & Finetuning pretrained PAR_v2_HPP2')
                self.pretrain_model = load_pretrain(encoder_type=self.encoder_type, trained_encoder=self.trained_encoder, use_vel=config.use_vel)
        elif self.encoder_type == "AR": self.pretrain_model = load_pretrain(encoder_type=self.encoder_type, trained_encoder=self.trained_encoder, use_vel=config.use_vel)
        elif self.encoder_type == "NAR": self.pretrain_model = load_pretrain(encoder_type=self.encoder_type, trained_encoder=self.trained_encoder, use_vel=config.use_vel)
        elif self.encoder_type == "NAT":
            print('Loading & Finetuning pretrained Neighborhood Attention')
            self.pretrain_model = NATTEN_Encoder(config,
                                                 window=config.model_config["natten_encoder_config"]["window"],
                                                 n_layers=config.model_config["natten_encoder_config"]["n_layers"],
                                                 n_unit=config.model_config["natten_encoder_config"]["n_unit"],
                                                 )

    def forward(self, label, audio, t, cond_drop_prob=None, save_encoder_features=False, saved_encoder_feature=None):
        # features (=cond_emb) : B x T*88 x H
        # label (=x_t) : B x T*88 x 1
        # features are concatanated with label as input to model
        batch = label.shape[0]
        if save_encoder_features == False or t[0].item()==self.diffusion_step-1:
            features_ = self.pretrain_model(audio, device=self.devices)
        else:
            assert th.is_tensor(saved_encoder_feature)
            features_ = saved_encoder_feature
        features = features_.transpose(-2, -1) # B x T x 88 x H(=128)
        features = features.reshape(features.shape[0], -1, features.shape[-1]) # B x T*88 x H
        if self.use_cfg == True:
            if cond_drop_prob != 1: cond_drop_prob = self.cond_drop_prob
            keep_mask = th.zeros((batch,1,1), device=self.devices).float().uniform_(0, 1) < (1 - cond_drop_prob)
            null_cond_emb = self.null_features_emb.repeat(label.shape[0], label.shape[1], 1) # B x T*88 x label_embed_dim
            features = th.where(keep_mask, features, null_cond_emb) # TODO - check if this is correct
        assert label.max() <= self.label_emb.num_embeddings - 1 and label.min() >= 0, f"Label out of range: {label.max()} {label.min()}"
        label_emb = self.label_emb(label) # B x T*88 x label_embed_dim
        input_features = th.cat((label_emb, features), dim=-1) # TODO - check if features shape = B x T*88 x (H + label_embed_dim)
        if self.cross_condition == 'self':
            x = self.trans_model(input_features, None, t)
        elif self.cross_condition == 'cross' or self.cross_condition == 'self_cross':
            x = self.trans_model(input_features, features, t)
        out = self.output(x)
        if save_encoder_features: return out.reshape(x.shape[0], x.shape[1]*x.shape[2], -1).permute(0, 2, 1), features_
        else:
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



class CNNModel(nn.Module):
    def __init__(self, config, use_vel=False):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        self.cnn_unit = config.cnn_unit
        self.middle_unit = config.middle_unit
        self.hidden_per_pitch = config.hidden_per_pitch
        self.pitchwise = config.pitchwise_lstm

        self.frontend = MIDIFrontEnd(5)
        self.block_1 = get_conv2d_block(3, self.cnn_unit, kernel_size=7)
        self.block_2 = get_conv2d_block(self.cnn_unit, self.cnn_unit, kernel_size=7)
        self.block_2_5 = get_conv2d_block(self.cnn_unit, self.cnn_unit, kernel_size=7, pool_size=(1,5))

        c3_out = 128

        self.conv_3 = HarmonicDilatedConv(self.cnn_unit, c3_out, 1)
        self.conv_4 = HarmonicDilatedConv(c3_out, c3_out, 1)
        self.conv_5 = HarmonicDilatedConv(c3_out, c3_out, 1)

        self.block_4 = get_conv2d_block(c3_out, c3_out, dilation=[1, 12], use_film=True, n_f=99)
        self.block_5 = get_conv2d_block(c3_out, c3_out, dilation=[1, 12], use_film=True, n_f=88)
        self.block_6 = get_conv2d_block(c3_out, c3_out, [5,1], use_film=True, n_f=88)
        self.block_7 = get_conv2d_block(c3_out, c3_out, [5,1], use_film=True, n_f=88)
        self.block_8 = get_conv2d_block(c3_out, c3_out, [5,1], use_film=True, n_f=88)

    def forward(self, audio, device=None):
        mel = self.frontend(audio)
        x = self.block_1(mel.permute(0, 1, 3, 2))
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.block_4(x)
        x = x[:,:,:,:88]
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)

        x = x.permute(0, 2, 1, 3)  # B, 128, T, 88 -> B, T, 128, 88
        return x


class HPPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        self.cnn_unit = config.cnn_unit
        self.middle_unit = config.middle_unit
        self.hidden_per_pitch = config.hidden_per_pitch
        self.pitchwise = config.pitchwise_lstm
        
        self.frontend = CQTFrontEnd()
        self.CNN = CNNTrunk()

    def forward(self, audio, device=None):
        cqt = self.frontend(audio).unsqueeze(1)
        x = self.CNN(cqt)  # B x C X T x 88
        x = x.permute(0, 2, 1, 3)  # B, 128, T, 88 -> B, T, 128, 88
        return x


class CQTFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        e = 2**(1/24)
        BINS_PER_SEMITONE=4
        self.to_cqt = nnAudio.Spectrogram.CQT(sr=SR, hop_length=HOP, fmin=27.5/e, n_bins=88*4, bins_per_octave=BINS_PER_SEMITONE*12, output_format='Magnitude')
        self.amplitude_to_db = transforms.AmplitudeToDB(top_db=80)

    def forward(self, audio):
        x = self.to_cqt(audio).float()[:,:,:-1]
        x = x.permute(0, 2, 1) # B, T, 352
        x = self.amplitude_to_db(x)
        return x


class HDC(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 48])
        self.conv_2 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 76])
        self.conv_3 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 96])
        self.conv_4 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 111])
        self.conv_5 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 124])
        self.conv_6 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 135])
        self.conv_7 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 144])
        self.conv_8 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 152])
    def forward(self, x):
        x = self.conv_1(x) + self.conv_2(x) + self.conv_3(x) + self.conv_4(x) +\
            self.conv_5(x) + self.conv_6(x) + self.conv_7(x) + self.conv_8(x)
        x = th.relu(x)
        return x

def get_conv2d_block_HPP(channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
    if(pool_size == None):
        return nn.Sequential( 
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ReLU(),
            # nn.BatchNorm2d(channel_out),
            nn.InstanceNorm2d(channel_out),
            
        )
    else:
        return nn.Sequential( 
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            # nn.BatchNorm2d(channel_out),
            nn.InstanceNorm2d(channel_out)
        )

class CNNTrunk(nn.Module):

    def __init__(self, c_in = 1, c_har = 16,  embedding = 128, fixed_dilation = 24) -> None:
        super().__init__()

        self.block_1 = get_conv2d_block_HPP(c_in, c_har, kernel_size=7)
        self.block_2 = get_conv2d_block_HPP(c_har, c_har, kernel_size=7)
        self.block_2_5 = get_conv2d_block_HPP(c_har, c_har, kernel_size=7)

        c3_out = embedding
        
        self.conv_3 = HDC(c_har, c3_out)

        self.block_4 = get_conv2d_block_HPP(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48])
        self.block_5 = get_conv2d_block_HPP(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = get_conv2d_block_HPP(c3_out, c3_out, [5,1])
        self.block_7 = get_conv2d_block_HPP(c3_out, c3_out, [5,1])
        self.block_8 = get_conv2d_block_HPP(c3_out, c3_out, [5,1])

    def forward(self, log_gram_db):
        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)

        x = self.block_5(x)
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x

        return x

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
    def __init__(self, hidden, config, window=25, n_unit=24, n_head=4, n_layers=2, cross_condition=False):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_unit = n_unit
        self.timestep_type= config.model_config['params']['timestep_type']
        self.diffusion_step= config.diffusion_config['params']['diffusion_step']
        self.lstm = nn.LSTM(hidden, n_unit//2, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.natten_dir = config.model_config['params']['natten_direction']
        self.spatial_size = config.model_config['label_emb_config']['spatial_size']
        self.cross_condition = cross_condition
        if self.natten_dir == '1d':
            self.na_1t = nn.ModuleList([NeighborhoodAttention1D_diffusion(n_unit, 1, window[0],
                                                                        spatial_size=self.spatial_size,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                    
                                    ] * n_layers)
            self.na_1f = nn.ModuleList([NeighborhoodAttention1D_diffusion(n_unit, 1, window[1],
                                                                        spatial_size=self.spatial_size,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type)
                                    ] * n_layers)
        elif self.natten_dir == '2d' and cross_condition == 'self':
            self.na = []
            for i in range(n_layers):
                self.na.append(NeighborhoodAttention2D_diffusion(n_unit, 4, window[i],
                                                                    diffusion_step=self.diffusion_step,
                                                                    dilation=config.model_config['lstm_natten_config']['dilation'][i],
                                                                    timestep_type=self.timestep_type))
            self.na = nn.ModuleList(self.na)
            # self.na = nn.ModuleList([NeighborhoodAttention2D_diffusion(n_unit, 4, window,
            #                                                             diffusion_step=self.diffusion_step,
            #                                                             timestep_type=self.timestep_type)] * n_layers)
        elif self.natten_dir == '2d' and cross_condition == 'cross':
            self.na = []
            for i in range(n_layers):
                self.na.append(NeighborhoodCrossAttention2D_diffusion(n_unit, 4, window[i],
                                                                    diffusion_step=self.diffusion_step,
                                                                    dilation=config.model_config['lstm_natten_config']['dilation'][i],
                                                                    timestep_type=self.timestep_type))
            self.na = nn.ModuleList(self.na)
        elif self.natten_dir == '2d' and cross_condition == 'self_cross':
            self.na = nn.ModuleList([NeighborhoodAttention2D_diffusion(n_unit, 4, window,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                    NeighborhoodCrossAttention2D_diffusion(n_unit, 4, window,
                                                                        diffusion_step=self.diffusion_step,
                                                                        timestep_type=self.timestep_type),
                                     ] * (n_layers // 2))


    def forward(self, x, cond=None, t=None):
        """
        x shape : B x T*88 x n_unit
        cond shape : B x T*88 x feature_embed_dim(=128)
        """
        B = x.shape[0]
        H = x.shape[-1]
        T = x.shape[1]//88
        x = x.reshape(B, T, 88, H) # B x T x 88 x H
        x = x.permute(0, 2, 1, 3).reshape(B*88, T, H) # B*88 x T x H
        x, c = self.lstm(x)

        # if use natten1d
        if self.natten_dir == '1d':
            for layer_1t, layer_1f in zip(self.na_1t, self.na_1f):
                # x_res = x
                x, t = layer_1t(x, t)
                # x = x + x_res
                x = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3).reshape(B*T, 88, self.n_unit)
                # x_res = x
                x, t = layer_1f(x, t)
                x = x.reshape(B, T, 88, self.n_unit).permute(0,2,1,3).reshape(B*88, T, self.n_unit)
                # x = x + x_res
            x = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3) # B x T x 88 x H
        
        # if use natten2d 
        # elif self.natten_dir == '2d' and self.cross_condition == 'cross':
        elif self.natten_dir == '2d':
            x = x.reshape(B, 88, T, -1).permute(0,2,1,3) # B x T x 88 x H
            for layers in self.na:
                x_res = x
                x, _, t = layers(x, cond, t)
                x = x + x_res

        # # if use natten2d
        # elif self.natten_dir == '2d' and not self.cross_condition == 'self':
        #     x = x.reshape(B, 88, T, -1).permute(0,2,1,3) # B x T x 88 x H
        #     for layers in self.na:
        #         x_res = x
        #         x, t = layers(x, t)
        #         x = x + x_res

        return x
    

class NATTEN_Encoder(nn.Module):
    def __init__(self, config, window=[5, 5, 25, 25], n_unit=24, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_unit = n_unit
        self.timestep_type= config.model_config['params']['timestep_type']
        self.natten_dir = config.model_config['params']['natten_direction']
        self.spatial_size = config.model_config['label_emb_config']['spatial_size']
        self.frontend_type = config.model_config['natten_encoder_config']['frontend_type']
        self.init_layer_type = config.model_config['natten_encoder_config']['init_layer_type']
        if self.frontend_type == 'midispec':
            self.frontend = MIDIFrontEnd(5)
            if self.init_layer_type == 'conv2d': self.block = get_conv2d_block_HPP(3, n_unit, kernel_size=7)
            elif self.init_layer_type == 'linear': self.block = nn.Linear(3, n_unit)
        elif self.frontend_type == 'cqt':
            self.frontend = CQTFrontEnd()
            if self.init_layer_type == 'conv2d': self.block = get_conv2d_block_HPP(1, n_unit, kernel_size=7)
            elif self.init_layer_type == 'linear': self.block = nn.Linear(1, n_unit)
        self.na = []
        for i in range(len(window)):
            self.na.append(NeighborhoodAttention2D_diffusion_encoder(n_unit, 4, window[i],
                                                                    diffusion_step=config.diffusion_config['params']['diffusion_step'],
                                                                    dilation=config.model_config['natten_encoder_config']['dilation'][i],
                                                                    timestep_type=self.timestep_type))
        self.na = nn.ModuleList(self.na)

    def forward(self, x, device=None):
        """
        x shape : B x T*88 x n_unit
        cond shape : B x T*88 x feature_embed_dim(=128)
        """

        B = x.shape[0]
        H = x.shape[-1]
        T = x.shape[1]//(4*88)
        if self.frontend_type == 'midispec':
            x = self.frontend(x)
        elif self.frontend_type == 'cqt':
            x = self.frontend(x).unsqueeze(1)
        if self.init_layer_type == 'conv2d':
            x = self.block(x)
            x = x.permute(0, 2, 3, 1)
        elif self.init_layer_type == 'linear':
            x = self.block(x.permute(0, 2, 3, 1))

        for i, layers in enumerate(self.na):
            if i < len(self.na)//2:
                x_res = x
                x = layers(x)
                x = x + x_res
                if i < 2:
                    x = F.max_pool2d(x, kernel_size=(2,1), stride=(2,1))

            if i >= len(self.na)//2:
                x_res = x
                x = layers(x)
                x = x + x_res

        return x.transpose(-2,-1) # B x T x 88 x H