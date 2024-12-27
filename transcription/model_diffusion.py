import torch as th
from torch import nn

from .diffusion.natten_diffusion import NeighborhoodAttention2D_diffusion, NeighborhoodAttention1D_diffusion, NeighborhoodCrossAttention2D_diffusion, NeighborhoodAttention2D_diffusion_encoder
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
        
        # Load pretrained model
        if self.encoder_type == "NAR": self.pretrain_model = load_pretrain(encoder_type=self.encoder_type, trained_encoder=self.trained_encoder, use_vel=config.use_vel)
        else:
            raise NotImplementedError(f"Encoder type {self.encoder_type} is not implemented")

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
            features = th.where(keep_mask, features, null_cond_emb) 
        assert label.max() <= self.label_emb.num_embeddings - 1 and label.min() >= 0, f"Label out of range: {label.max()} {label.min()}"
        label_emb = self.label_emb(label) # B x T*88 x label_embed_dim
        input_features = th.cat((label_emb, features), dim=-1) 
        if self.cross_condition == 'self':
            x = self.trans_model(input_features, None, t)
        elif self.cross_condition == 'cross' or self.cross_condition == 'self_cross':
            x = self.trans_model(input_features, features, t)
        out = self.output(x)
        if save_encoder_features: return out.reshape(x.shape[0], x.shape[1]*x.shape[2], -1).permute(0, 2, 1), features_
        else:
            return out.reshape(x.shape[0], x.shape[1]*x.shape[2], -1).permute(0, 2, 1) # B x 5 x T*88


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