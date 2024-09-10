# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F
import importlib

# from image_synthesis.utils.misc import instantiate_from_config
import numpy as np

from inspect import isfunction
from torch.cuda.amp import autocast
from termcolor import colored
eps = 1e-8

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    # a = log(a') -> return log(1 - a')
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    # a = log(a'), b = log(b') -> return log(a' + b')
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_matmul(log_a, log_b):
    # a = log(a'), b = log(b') -> return log(a' @ b')
    return torch.log(torch.einsum('bij,bjk->bik', torch.exp(log_b), torch.exp(log_a)))
    # return log_a.unsqueeze(-1) + log_b.unsqueeze(-3)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(0, t.unsqueeze(-1).unsqueeze(-1).expand(b, a.shape[1], a.shape[2]))
    # return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return out

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # (1, 1024) -> (1, 1024, 2888)
    permute_order = (0, -1) + tuple(range(1, len(x.size()))) # 0, -1, 1
    x_onehot = x_onehot.permute(permute_order) # (1, 1024, 2888) -> (1, 2888, 1024)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

BASE = ['off', 'offset', 'onset', 'sustain', 'reonset', 'MASK']
def get_custom_transition_mat(at, bt, ct, t, r=0.0):
    b = bt[t]
    c = ct[t]
    a = at[t]
    mat  =  [[b+a-5*r*b , b   , b   , b , b    , 0.0],
             [(r+1)*b, b+a, b   , b   , b     , 0.0],
             [(r+1)*b, b  , b+a , b   , b     , 0.0],
             [(r+1)*b, b  , b   , b+a , b  , 0.0],
             [(r+1)*b, b  , b   , b   , b+a  , 0.0],
             [c+r*b  , c    , c    , c    , c    , 1.0],
             ]
    return np.array(mat, dtype=np.float64)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # accumulate (0.9999 ~ 0)
    att = np.concatenate(([1], att)) 
    at = att[1:]/att[:-1] # alpha (no change prob)
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # accumulate (0 ~ 0.9999)
    ctt = np.concatenate(([0], ctt)) # this is \bar{gamma}
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct # Think this is gamma (random mask prob)
    bt = (1-at-ct)/N # Think this is beta (random resample prob)
    att = np.concatenate((att[1:], [1])) # last 1 is for mask
    ctt = np.concatenate((ctt[1:], [0])) # last 0 is for mask
    btt = (1-att-ctt)/N # Think this is \bar{beta}
    return at, bt, ct, att, btt, ctt

def get_alpha_schedules(num_steps, N=5, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.linspace(att_1, att_T, num_steps+1, dtype=np.float64) # accumulate (0.9999 ~ 0)
    at = att[1:]/att[:-1] # alpha (no change prob)
    ctt = np.linspace(ctt_1, ctt_T, num_steps+1, dtype=np.float64) # accumulate (0.9999 ~ 0)
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct # Think this is gamma (random mask prob)
    bt = (1-at-ct)/N # Think this is beta (random resample prob)
    # calculate btt from bt
    btt = np.zeros_like(att)
    btt = (1-att-ctt)/N # Think this is beta (random resample prob)
    # print(btt)
    return at, bt, ct, att, btt, ctt

def get_hounsu_powerschedules(time_step, N=5, at_1 = 0.99999, at_T = 0.70, ct_1 = 0.000009, ct_T = 0.299999, a_pow=2.4, c_pow=2.70): # beta max 0.02, ct schedule 대칭
    # obtain btt from bt_1, bt_T
    at = (np.linspace(0, 1, time_step, dtype=np.float64)**a_pow)*(at_T - at_1) + at_1 # accumulate (0.9999 ~ 0)
    att = np.zeros((at.shape[0]+1,), dtype=np.float64)
    att[0] = at[0]
    for i in range(1, len(att)):
        att[i] = att[i-1] * at[i-1]
    
    # obtain ctt from ct_1, ct_T
    ct = (np.linspace(0, 1, time_step, dtype=np.float64)**c_pow)*(ct_T - ct_1) + ct_1 # accumulate (0.9999 ~ 0)
    ctt = np.zeros((at.shape[0]+1,), dtype=np.float64)
    ctt[0] = ct[0]
    prod = 1 - ct[0]
    for i in range(1, len(ctt)):
        prod = prod * (1-ct[i-1])
        ctt[i] = 1 - prod
    bt = (1-ct-at)/N
    btt = (1-ctt-att)/N
    return at, bt, ct, att, btt, ctt

def get_hounsu_unbalanced_schedules(time_step, alpha, N=5, r=0.0):  # beta max 0.02, ct schedule 대칭
    if alpha == 'alpha1':
        at, bt, ct, att, btt, ctt = alpha_schedule(time_step+1, N=N, ctt_T=0.99999)
    elif alpha == 'alpha2':
        at, bt, ct, att, btt, ctt = alpha_schedule(time_step+1, N=N, ctt_T=0.9)
    elif alpha == 'alpha3':
        at, bt, ct, att, btt, ctt = get_alpha_schedules(time_step+1, N=N, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999)
    elif alpha == 'alpha4':
        at, bt, ct, att, btt, ctt = get_alpha_schedules(time_step+1, N=N, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.9)
    elif alpha == 'alpha5':
        at, bt, ct, att, btt, ctt = get_hounsu_powerschedules(time_step+1, N=N)
    transitions = np.zeros((time_step+1, N+1, N+1), np.float64)
    products = np.zeros((time_step+2, N+1, N+1), np.float64)
    mat = np.eye(N+1, dtype=np.float64)
    for t in range(time_step+1):
        transition = get_custom_transition_mat(at, bt, ct, t, r=r)
        transitions[t] = transition
        # transition = get_gaussian_transition_mat(t, dim=dim - 1)
        # transitions[t, :dim - 1, :dim - 1] = (1 - beta) * transition
        # transitions[t, -1] = beta
        # transitions[t, -1, -1] = 1.
        mat = transitions[t] @ mat
        products[t] = mat
    return transitions, products

class DiscreteDiffusionCustomized(nn.Module):
    def __init__(
        self,
        *,
        model,
        config,
        device,
        diffusion_step=100,
        alpha_init_type='cos',
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        mask_weight=[1,1],
    ):
        super().__init__()

        # self.condition_emb = 
        # self.condition_dim = self.condition_emb.embed_dim
       
        self.model = model
        self.label_seq_len = config.model_config['params']['label_seq_len']
        self.amp = False

        self.num_classes = 6
        self.loss_type = 'vb_stochastic'
        self.shape = config.model_config['params']['label_seq_len']
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        self.devices = device
        if config.diffusion_config["params"]["onset_suppress_sample"]:
            self.onset_suppress = config.diffusion_config["params"]["onset_suppress_sample"]
        else : self.onset_suppress = False
        if config.diffusion_config["params"]["onset_weight_kl"]:
            self.onset_weight_kl = config.diffusion_config["params"]["onset_weight_kl"]
        else : self.onset_weight_kl = False
        if config.diffusion_config["params"]["classifier_free_guidance"]:
            self.cond_scale = config.diffusion_config["params"]["classifier_free_guidance"]["cond_scale"]
        else:
            self.cond_scale = 1
        
        # build precomputed customized transition matrices, disregard mask token for now(=self.num_classes-1)
        print(colored("Creating precomputed customized discrete diffusion transition matrices", 'green', attrs=['bold']))

        transitions, products = get_hounsu_unbalanced_schedules(self.num_timesteps, alpha=alpha_init_type, N=self.num_classes-1, r=config.diffusion_config["params"]["r_value"])

        transitions = torch.tensor(transitions, dtype=torch.float64)
        products = torch.tensor(products, dtype=torch.float64)
        log_transitions = torch.log(transitions.clamp(min=1e-30))
        log_products = torch.log(products.clamp(min=1e-30))
        # we don't need log conversion of transitions and products because there are only K=5 tokens (O(K^2))
        # at = torch.tensor(at.astype('float64')) # (t,)
        # bt = torch.tensor(bt.astype('float64')) # (t,)
        # ct = torch.tensor(ct.astype('float64')) # (t,)
        # log_at = torch.log(at) # (t,)
        # log_bt = torch.log(bt) # (t,)
        # log_ct = torch.log(ct) # (t,)
        # att = torch.tensor(att.astype('float64')) # (t+1,)
        # btt = torch.tensor(btt.astype('float64')) # (t+1,)
        # ctt = torch.tensor(ctt.astype('float64')) # (t+1,)
        # log_cumprod_at = torch.log(att) # (t+1,)
        # log_cumprod_bt = torch.log(btt) # (t+1,)
        # log_cumprod_ct = torch.log(ctt) # (t+1,)

        # log_1_min_ct = log_1_min_a(log_ct)
        # log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        # assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        # assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        # self.register_buffer('log_at', log_at.float())
        # self.register_buffer('log_bt', log_bt.float())
        # self.register_buffer('log_ct', log_ct.float())
        # self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        # self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        # self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        # self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        # self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('transitions', transitions.float())
        self.register_buffer('products', products.float())
        self.register_buffer('log_transitions', log_transitions.float())
        self.register_buffer('log_products', log_products.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None


    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        if self.onset_weight_kl:
            kl = (log_prob1.exp() * (log_prob1 - log_prob2))
            kl[:, [2, 4], :] *= (1 + self.onset_weight_kl)  # Apply onset_weight_kl factor to indices 2 and 4
            kl = kl.sum(dim=1)
        else:
            kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl


    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_products = extract(self.log_products, t, log_x_t.shape)             # at
        log_probs = log_matmul(log_x_t, log_products)
        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        '''
        x_start shape : (B, 6(=one-hot token), 1024(=label_len))
        '''
        # log_x_start can be onehot or not
        # if t[0] == -1:
        #     print(t)
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_transitions = extract(self.log_transitions, t, log_x_start.shape)
        log_probs = log_matmul(log_x_start, log_transitions)

        return log_probs

    def predict_start(self, log_x_t, cond_audio, t, sampling=False):          # p(x0|xt)
        x_t_idx = log_onehot_to_index(log_x_t) # idx is further embedded to higher dimension
        if sampling==False:
            if self.amp == True:
                with autocast():
                    out = self.model(x_t_idx, cond_audio, t)
            else:
                out = self.model(x_t_idx, cond_audio, t)
        if sampling==True:
            if self.amp == True:
                with autocast():
                    out = self.model(x_t_idx, cond_audio, t)
                    if self.cond_scale != 1:
                        null_out = self.model(x_t_idx, cond_audio, t, cond_drop_prob = 1.)
                        out = out * self.cond_scale + null_out * (1 - self.cond_scale)
                        
            else:
                out = self.model(x_t_idx, cond_audio, t)
                if self.cond_scale != 1:
                    null_out = self.model(x_t_idx, cond_audio, t, cond_drop_prob = 1.)
                    out = out * self.cond_scale + null_out * (1 - self.cond_scale)

        assert out.size(0) == x_t_idx.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t_idx.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float() # softmax then log B x 5 x T*88
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.label_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.label_seq_len) # B x 1 x 27544 vector filled with -69.08

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        log_qt = log_qt[:,:-1,:] 
        # log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        # ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # log_qt = (~mask)*log_qt + mask*ct_cumprod_vector # replace the masked tokens's probability to only select mask token afterwards
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        # log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        # ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        # ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        # log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1) # B x 6 x 27544
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True) # B x 1 x 27544
        q = q - q_log_sum_exp # normalize q
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_audio, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        # p_pred for sampling
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, cond_audio, t, sampling=True)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_audio, t, sampling=True)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, log_x, cond_audio, t, cond_drop_prob=None):               # sample q(xt-1) for next step from xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, cond_audio, t) # TODO : onset, reonset -> 2, 4 (0~4)
        if self.onset_suppress: # suppress onset, offsets when sampling
            model_log_prob[:, 2, :] = model_log_prob[:, 2, :] * (1 + self.onset_suppress)
            model_log_prob[:, 4, :] = model_log_prob[:, 4, :] * (1 + self.onset_suppress)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits) # gaussian distribution of mean 0 and variance 1
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_audio, is_train=True):                       # get the KL loss
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')

        # DALLE encodes image to 2887+1(mask) tokens (=self.num_classes)
        log_x_start = index_to_log_onehot(x_start, self.num_classes) # but why log?
        # x0 -> xt
        log_xt = self.q_sample(log_x_start=log_x_start, t=t) # B x 6 x L , discretely-noised log one-hot vector
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_audio, t=t)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon) # start prediction
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob) # xt-1 prediction
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes-1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, (vb_loss, kl, decoder_nll, loss1, loss2)


    @property
    def device(self):
        return self.model.to_logits[-1].weight.device

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'label_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.model.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self, 
            label,
            features, 
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            is_train=True,
            **kwargs):
        """
        input shape : B x T*88 x H+1 
        """
        if kwargs.get('autocast') == True:
            self.amp = True
        batch_size = label.shape[0]
        device = label.device

        # if self.condition_emb is not None:
        #     with autocast(enabled=False):
        #         with torch.no_grad():
        #             cond_emb = self.condition_emb(input['condition_token']) # (B, Ld(=77), D(=512)) # 256*1024
        #         cond_emb = cond_emb.float()
        # else: # share condition embeding with label
        #     if input.get('condition_embed_token') == None:
        #         cond_emb = None
        #     else:
        #         cond_emb = input['condition_embed_token'].float()
        cond_emb = None # no condition for now
            
        # now we get cond_emb and sample_image
        ######## Diffusion Training ########
        if is_train == True:
            log_model_prob, (loss, kl, decoder_nll, loss1, loss2) = self._train_loss(label, features)
            loss = loss.sum()/(label.size()[0] * label.size()[1]) # TODO: check if input.size()[0] and [1] is right
            kl = kl.sum()/(label.size()[0] * label.size()[1])
            decoder_nll = decoder_nll.sum()/(label.size()[0] * label.size()[1])
            loss1 = loss1.sum()/(label.size()[0] * label.size()[1])
            loss2 = loss2.sum()/(label.size()[0] * label.size()[1])

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss 
            out['kl'] = kl
            out['decoder_nll'] = decoder_nll
            out['loss1'] = loss1
            out['loss2'] = loss2
        self.amp = False
        return out


    def sample(
            self,
            audio,
            filter_ratio = 0.5,
            temperature = 1.0,
            return_att_weight = False,
            return_logits = False,
            label_logits = None,
            print_log = True,
            visualize_denoising = False,
            **kwargs):

        batch_size = audio.shape[0]
    
        device = self.log_transitions.device
        start_step = int(self.num_timesteps * filter_ratio)

        cond_audio = audio # no condition for now
        labels = [] # for plotting labels getting denoised

        if start_step == 0:
            # use full mask sample
            zero_logits = torch.zeros((batch_size, self.num_classes-1, self.shape),device=device)
            one_logits = torch.ones((batch_size, 1, self.shape),device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1) # B, class, token_len
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in range(start_step-1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long) # make batch tensor filled with value 't'
                    log_z = self.p_sample(log_z, cond_audio, t)     # log_z is log_onehot
                    if visualize_denoising and diffusion_index % 10 == 0:
                        labels.append(log_z.argmax(1).cpu().numpy())

        else: # TODO : fill this. check whether there are no mistake in VQ diffusion
            raise ValueError("Not implemented yet, but what is this for anyway?")
        

        label_token = log_onehot_to_index(log_z)
        
        output = {'label_token': label_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output, labels