import heapq
import argparse
import os
import sys
import tempfile
from pathlib import Path
import json
import csv
from types import SimpleNamespace
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import time

import torch as th
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
# from torch.nn.utils import clip_grad_norm_
from .diffusion.clip_grad_norm import ClipGradNorm
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP

from transcription.diffusion.trainer import DiscreteDiffusion
from termcolor import colored

import numpy as np
import wandb

# from adabelief_pytorch import AdaBelief

from .diffusion.lr_scheduler import ReduceLROnPlateauWithWarmup
from .model_diffusion import TransModel
from .solver_diffusion import Solver
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP = True
except:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False
from .diffusion.ema import EMA
from .constants import HOP
from .data import MAESTRO_V3, MAESTRO, MAPS, EmotionDataset, SMD, ViennaCorpus
from .loss import FocalLoss
from .evaluate import evaluate
from .utils import summary, CustomSampler

th.autograd.set_detect_anomaly(True)

def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)

default_config = dict(
    n_mels=495, # 700?
    n_fft=4096,
    f_min=27.5,
    f_max=8000,
    cnn_unit=48,
    lstm_unit=48,
    hidden_per_pitch=128,
    n_per_pitch=5,
    frontend_kernel_size=7,
    fc_unit=768,
    shrink_channels=[4,1],
    batch_size=8,
    pitchwise_lstm=True,
    use_film=True,
    win_fw=4,
    win_bw=0,
    model='DiscDiff',
    dataset='MAESTRO_V3',
    seq_len=160256,
    n_workers=8,
    lr=5e-4,
    n_epoch = 100,
    noisy_condition=False,
    valid_interval=5000,
    valid_seq_len=160256, # 716800
    enhanced_context=True,
    multifc=True,
    cnn_widths = [3,3,3,3,3,3],
    debug=False,
    seed=1000,
    resume_dir=None,
    iteration=1000000,
    port=23456
    )
   
   
def setup(rank, world_size, port=23456):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                        init_method=f'tcp://127.0.0.1:{port}',
                        world_size=world_size,
                        rank=rank)

def cleanup():
    dist.destroy_process_group()
    
def get_dataset(config, split, sample_len=160256, random_sample=False, transform=False, load_mode='lazy'):
    if config.dataset == 'MAESTRO_V3':
        return MAESTRO_V3(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform)
    elif config.dataset == 'MAESTRO_V1':
        return MAESTRO(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform)
    elif config.dataset == 'MAPS':
        return MAPS(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform)
    elif config.dataset == 'Emotion':
        return EmotionDataset(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform)
    elif config.dataset == 'SMD':
        return SMD(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform)
    elif config.dataset == 'Vienna':
        return ViennaCorpus(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform)
    else:
        raise KeyError

class ModelSaver():
    def __init__(self, config, order='lower', n_keep=5, resume=False):
        self.logdir = Path(config.logdir)
        self.order = order
        assert order in ['lower', 'higher']
        self.config = config
        self.n_keep = n_keep
        self.top_n = []
        self.best_ckp = None
        self.last_ckp = None
        self.last_opt = None
        self.last_sch = None
        self.last_step = -1

        if resume:
            with open(self.logdir / 'checkpoint.csv', "r") as f:
                reader = csv.reader(f, delimiter=',')
                self.top_n = [(el[0], float(el[1]), int(el[2])) for el in list(reader)]
            self.best_ckp = self.top_n[0][0]
            lastest = np.argmax([el[2] for el in self.top_n])
            self.last_ckp = self.top_n[lastest][0]
            self.last_step = self.top_n[lastest][2]
            self.last_opt = self.save_name_opt(self.last_step)
            self.last_sch = self.save_name_sch(self.last_step)

    def save_model(self, trainer, ema, save_name, ddp):
        save_dict = self.config.__dict__
        state_dict = trainer.model.module.state_dict() if ddp else trainer.model.state_dict()
        save_dict['model_state_dict'] = state_dict
        if ema is not None:
            state_dict = ema.module.state_dict() if ddp else ema.state_dict() # TODO: check for the ddp case
            save_dict['ema_state_dict'] = state_dict
        th.save(save_dict, self.logdir / save_name)
        self.last_ckp = save_name

    def update_optim(self, optimizer, step):
        opt_name = self.save_name_opt(step)
        th.save(optimizer.state_dict(), self.logdir / opt_name)
        last_opt = self.logdir / self.save_name_opt(self.last_step)
        if last_opt.exists():
            last_opt.unlink()
        self.last_opt = opt_name

    def save_name_opt(self, step):
        if step > 1000:
            return f'opt_{step//1000}k.pt'
        else:
            return f'opt_{step}.pt'

    def update_sch(self, scheduler, step):
        sch_name = self.save_name_sch(step)
        th.save(scheduler.state_dict(), self.logdir / sch_name)
        last_sch = self.logdir / self.save_name_sch(self.last_step)
        if last_sch.exists():
            last_sch.unlink()
        self.last_sch = sch_name

    def save_name_sch(self, step):
        if step > 1000:
            return f'sch_{step//1000}k.pt'
        else:
            return f'sch_{step}.pt'
    
    def save_name(self, step, score):
        if step > 1000:
            return f'model_{step//1000}k_{score:.4f}.pt'
        else:
            return f'model_{step}_{score:.4f}.pt'
    
    def write_csv(self):
        with open(self.logdir / 'checkpoint.csv', "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([(el[0], el[1], el[2]) for el in self.top_n])
    
    def update(self, trainer, ema, optimizer, scheduler, step, score, ddp):
        save_name = self.save_name(step, score)
        self.save_model(trainer, ema, save_name, ddp)
        self.update_optim(optimizer, step)
        self.update_sch(scheduler, step)
        self.top_n.append((save_name, score, step))
        self.update_top_n()
        self.write_csv()
        self.last_step = step

    def update_top_n(self): 
        if self.order == 'lower':
            reverse = False
        elif self.order == 'higher':
            reverse = True
        self.top_n.sort(key=lambda x: x[1], reverse=reverse)
        self.best_ckp = self.top_n[0][0]
        if len(self.top_n) > self.n_keep:
            del_list = self.top_n[self.n_keep:]
            self.top_n = self.top_n[:self.n_keep]
            for save_name, score, step in del_list:
                if self.last_ckp == save_name:
                    self.top_n.append((save_name, score, step))
                    continue
                (self.logdir / save_name).unlink()

def sample_func(trainer, audio, ema, config, visualize_denoising=False):
    tic = time.time()
    # print('\nBegin to sample...\n')
    if ema is not None:
        ema.modify_to_inference()
        suffix = '_ema'
    else:
        suffix = ''
    with th.no_grad():
        if config.debug == False:
            if config.amp:
                with autocast(): # TODO : work on trainer.py
                    samples, labels = trainer.sample(audio,
                                             filter_ratio = 0, # TODO : what is this for?
                                             visualize_denoising=visualize_denoising
                                             ) # input : [features + noisy label]
            else:
                samples, labels = trainer.sample(audio, 
                                         filter_ratio=0,
                                         visualize_denoising=visualize_denoising
                                         )
        else:
            samples, labels = trainer.sample(audio,
                                     filter_ratio=0,
                                     visualize_denoising=visualize_denoising
                                     ) 
    
        if ema is not None:
            ema.modify_to_train()
    
    # print('Sample done, time: {:.2f}'.format(time.time() - tic)) 
    return samples['label_token'], labels
    
def train_step(model, batch, ema, optimizer, scheduler, scaler, clip_grad_norm, step, device, config):
    audio = batch['audio'].to(device) # B x L
    label = batch['label'][:,1:,:].to(device) # B x T x 88
    # _ = batch['velocity'].to(device)

    # forward
    if config.amp:
        with autocast():
            label = label.reshape(label.shape[0], -1) # B x T*88
            disc_diffusion_loss = model(label, audio, return_loss=True) 
    else:
        label = label.reshape(label.shape[0], -1) # B x T*88
        disc_diffusion_loss = model(label, audio, return_loss=True)
                
    # backpropagate and update optimizer
    if config.amp:
        scaler.scale(disc_diffusion_loss['loss']).backward()
        if clip_grad_norm is not None:
            clip_grad_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()
    else:
        disc_diffusion_loss['loss'].backward()
        if clip_grad_norm is not None:
            clip_grad_norm(model.parameters())
        optimizer.step()
        optimizer.zero_grad()
    
    # update scheduler
    if scheduler is not None:
        if config.solver["optimizers_and_schedulers"][0]['scheduler']['step_iteration'] > 0 and (step + 1) % config.solver['optimizers_and_schedulers'][0]['scheduler']['step_iteration'] == 0:
            scheduler.step(disc_diffusion_loss['loss']) 

    # update ema model
    if ema is not None:
        ema.update(iteration=step) # TODO: check if step == last_iter
    
    return disc_diffusion_loss,  None,  # vel_loss

def valid_step(trainer, batch, ema, step, device, config):
    audio = batch['audio'].to(device)
    label = batch['label'][:,1:,:].to(device)
    # _ = batch['velocity'].to(device)
    shape = label.shape
    # Forward step (for loss calculation)
    if config.amp:
        with autocast():
            label = label.reshape(label.shape[0], -1) # B x T*88
            disc_diffusion_loss = trainer(label, audio, return_loss=True) 
    else:
        label = label.reshape(label.shape[0], -1) # B x T*88
        disc_diffusion_loss = trainer(label, audio, return_loss=True)
    # Sample step (for metrics calculation)
    # if step % config.inference_step == 0:
    frame_out, _ = sample_func(trainer, audio, ema, config)
    # frame out: B x T x 88 x C
    accuracy = (frame_out == label).float()
    validation_metric = defaultdict(list)
    for n in range(audio.shape[0]):
        sample = frame_out.reshape(*shape)[n] 
        metrics = evaluate(sample, label.reshape(*shape)[n], band_eval=False)
        for k, v in metrics.items():
            validation_metric[k].append(v)
    validation_metric['accuracy_loss'] = accuracy.mean(dim=-1)
    validation_metric['disc_diffusion_loss'] = disc_diffusion_loss['loss'].unsqueeze(0)
    
    return validation_metric, frame_out

def test_step(trainer, batch, ema, step, device, config):
    audio = batch['audio'].to(device)
    B, audio_len = audio.shape[0], audio.shape[1]
    test_metric = defaultdict(list)

    frame_outs = [] 
    n_step = (audio_len - 1) // HOP + 1
    shape = (audio.shape[0], n_step, 88)
    seg_len = 313
    overlap = 50

    n_seg = (n_step - overlap) // (seg_len - overlap) + 1

    frame_outs = th.zeros(shape, dtype=th.int)

    for seg in tqdm(range(n_seg)):
    # for seg in tqdm(range(2)):
        start = seg * (seg_len - overlap)
        end = start + seg_len
        if end > n_step:
            audio_seg = audio[:, int(start*HOP):]
            audio_len = audio_seg.shape[1]
            # audio_pad = F.pad(audio_seg, (0, seg_len*HOP - audio_len))
            audio_pad = audio_seg # no padding
            audio_pad = audio_pad.to(device)
        else:
            audio_pad = audio[:, int(start*HOP):int(end*HOP)].to(device)

        frame_out, _ = sample_func(trainer, audio_pad, ema, config)
        sample = frame_out.reshape(frame_out.shape[0], -1, 88).detach() # B x T x 88
        if seg == 0:
            frame_outs[:, :end-overlap//2] = sample[:, :-overlap//2]
        elif seg == n_seg - 1:
            frame_outs[:, start+overlap//2:] = sample[:, overlap//2:n_step-start]
        else:
            frame_outs[:, start+overlap//2:end-overlap//2] = sample[:, overlap//2:-overlap//2]

    # frame out: B x T x 88 x C
    # vel_outs = []
    # obtain metrics on full-length audio for batches
    for n in range(batch['audio'].shape[0]):
        step_len = batch['step_len'][n]
        frame = sample[n][:step_len].detach().cpu()
        metrics = evaluate(frame_outs[n], batch['label'][n][1:].detach().cpu()[:-1], band_eval=False) # TODO: check if [:-1] is right.
        for k, v in metrics.items():
            test_metric[k].append(v)
        print(f'note f1: {metrics["metric/note/f1"][0]:.4f}, note_with_offsets_f1: {metrics["metric/note-with-offsets/f1"][0]:.4f}', batch['path'][n])
    
    return test_metric, frame_outs

class PadCollate:
    def __call__(self, data):
        max_len = data[0]['audio'].shape[0] // HOP
        
        for datum in data:
            step_len = datum['audio'].shape[0] // HOP
            datum['step_len'] = step_len
            pad_len = max_len - step_len
            pad_len_sample = pad_len * HOP
            datum['audio'] = F.pad(datum['audio'], (0, pad_len_sample))

        batch = defaultdict(list)
        for key in data[0].keys():
            if key == 'audio':
                batch[key] = th.stack([datum[key] for datum in data], 0)
            else :
                batch[key] = [datum[key] for datum in data]
        return batch


def train(rank, world_size, config, ddp=True):
    th.cuda.set_device(rank)
    if ddp:
        setup(rank, world_size, port=config.port)
    else:
        assert world_size == 1 and rank == 0
    device = f'cuda:{rank}'
    seed = config.seed + rank
    th.manual_seed(seed)
    np.random.seed(seed)

    # Load diffusion model and trainer
    model = TransModel(config, device=device)
    if not config.finetune:
        print(colored("Freezing pretrained model", "blue", attrs=["bold"]))
        for param in model.pretrain_model.parameters(): param.requires_grad = False
    elif config.finetune:
        print(colored("Finetuning/Training pretrained model", "red", attrs=["bold"]))
    if config.diffusion_config["params"]["customized_transition_matrix"]:
        from transcription.diffusion.trainer_customize import DiscreteDiffusionCustomized
        trainer = DiscreteDiffusionCustomized(
                        model=model,
                        config=config,
                        device=device,
                        diffusion_step=config.diffusion_config["params"]["diffusion_step"],
                        alpha_init_type=config.diffusion_config["params"]["alpha_init_type"],
                        auxiliary_loss_weight=config.diffusion_config["params"]["auxiliary_loss_weight"],
                        adaptive_auxiliary_loss=config.diffusion_config["params"]["adaptive_auxiliary_loss"],
                        mask_weight=config.diffusion_config["params"]["mask_weight"],
        )
    elif not config.diffusion_config["params"]["customized_transition_matrix"]:
        trainer = DiscreteDiffusion(
                        model=model,
                        config=config,
                        device=device,
                        diffusion_step=config.diffusion_config["params"]["diffusion_step"],
                        alpha_init_type=config.diffusion_config["params"]["alpha_init_type"],
                        auxiliary_loss_weight=config.diffusion_config["params"]["auxiliary_loss_weight"],
                        adaptive_auxiliary_loss=config.diffusion_config["params"]["adaptive_auxiliary_loss"],
                        mask_weight=config.diffusion_config["params"]["mask_weight"],
                        )
    trainer.to(rank)
    # count number of parameters in model using numel
    print(colored(f'pretrain_model parameters : {sum(p.numel() for p in trainer.model.pretrain_model.parameters())}', color='red', attrs=['bold']))
    print(colored(f'trans_model parameters : {sum(p.numel() for p in trainer.model.trans_model.parameters())}', color='green', attrs=['bold']))
    
    # trainer.model.pretrain_model.

    clip_grad_norm = ClipGradNorm(start_iteration=config.solver["clip_grad_norm"]["params"]["start_iteration"],
                                end_iteration=config.solver["clip_grad_norm"]["params"]["end_iteration"],
                                max_norm=config.solver["clip_grad_norm"]["params"]["max_norm"])

    # configure for ema 
    if 'ema' in config.solver and rank==0:
        ema_args = config.solver['ema']
        ema_args['model'] = trainer.model
        ema = EMA(**ema_args)
    else:
        ema = None
    # configure for amp
    if config.amp:
        scaler = GradScaler()
        print('Using AMP for training!')
    else : scaler = None
    # resume model
    if config.resume_dir:
        model_saver = ModelSaver(config, resume=True, order='higher')
        step = model_saver.last_step
    else:  
        model_saver = ModelSaver(config, order='higher')
        step = 0
    # wandb initialization
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if rank == 0:
        if not config.wandb: os.environ["WANDB_DISABLED"] = "true"
        if config.resume_dir:
            run = wandb.init('transcription', name=f'{config.id}_{time}', id=config.id, dir=config.logdir, resume="allow")
        else:   
            run = wandb.init('transcription', config=config, id=config.id, name=f'{config.name}_{time}', dir=config.logdir, resume="allow")
        summary(trainer)
    # configure for ddp
    if ddp:
        trainer = th.nn.SyncBatchNorm.convert_sync_batchnorm(trainer)
        trainer = DDP(trainer, device_ids=[rank])
    # optimizer = AdaBelief(model.parameters(), lr=config.lr, 
    #                       eps=1e-16, betas=(0.9,0.999), weight_decouple=True, 
    #                       rectify = False, print_change_log=False)
    optimizer = optim.AdamW(trainer.model.parameters(), lr=config.lr, 
                          betas=config.solver["optimizers_and_schedulers"][0]["optimizer"]["params"]["betas"],
                          weight_decay=config.solver["optimizers_and_schedulers"][0]["optimizer"]["params"]["weight_decay"])
    scheduler = ReduceLROnPlateauWithWarmup(optimizer,
                                            factor=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["factor"],
                                            patience=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["patience"],
                                            min_lr=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["min_lr"],
                                            threshold=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["threshold"],
                                            threshold_mode=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["threshold_mode"],
                                            warmup_lr=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["warmup_lr"],
                                            warmup=config.solver["optimizers_and_schedulers"][0]["scheduler"]["params"]["warmup"],)
    
    # resume checkpoint
    if config.resume_dir:
        if config.eval:
            ckp = th.load(model_saver.logdir / model_saver.best_ckp, map_location={'cuda:0':f'cuda:{rank}'})
        else:
            ckp = th.load(model_saver.logdir / model_saver.last_ckp, map_location={'cuda:0':f'cuda:{rank}'})
        if ddp:
            trainer.model.module.load_state_dict(ckp['model_state_dict'])
            # TODO : add loading  ema
        else:
            trainer.model.load_state_dict(ckp['model_state_dict'])
            ema.load_state_dict(ckp['ema_state_dict'])
        del ckp

        if not config.eval:
            ckp_opt = th.load(model_saver.logdir / model_saver.last_opt)
            ckp_sch = th.load(model_saver.logdir / model_saver.last_sch)
            # TODO : load optimizer, scheduler, ema, last_epoch, last_iter, clip_grad_norm
            optimizer.load_state_dict(ckp_opt)
            scheduler.load_state_dict(ckp_sch)
            del ckp_opt, ckp_sch
        if ddp:
            dist.barrier()

    # Print amount of trainable parameters
    total_params_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(colored(f"Total trainable parameters: {total_params_trainable}", "green", attrs=["bold"]))
        
    if not config.eval:
        if rank == 0:
            run.watch(trainer.model, log_freq=1000)

        train_set = get_dataset(config, ['train'], sample_len=config.seq_len, 
                                random_sample=True, transform=config.noisy_condition, load_mode='lazy')
        valid_set = get_dataset(config, ['validation'], sample_len=config.valid_seq_len,
                                random_sample=False, transform=False, load_mode='lazy')
        if ddp:
            train_sampler = DistributedSampler(dataset=train_set, num_replicas=world_size, 
                                            rank=rank, shuffle=True)
            segments = np.split(np.arange(len(valid_set)),
                                np.arange(len(valid_set), step=config.batch_size//world_size))[1:]  # the first segment is []
            target_segments = [el for n, el in enumerate(segments) if n%world_size == rank]
            valid_sampler = CustomSampler(target_segments)
            data_loader_valid = DataLoader(
                valid_set, batch_sampler=valid_sampler,
                num_workers=config.n_workers,
                pin_memory=False,
            )
        else:
            train_sampler=None
            data_loader_valid = DataLoader(
                valid_set, sampler=None,
                batch_size=config.batch_size,
                num_workers=config.n_workers,
                pin_memory=False,
            )
        data_loader_train = DataLoader(
            train_set, sampler=train_sampler,
            batch_size=config.batch_size//world_size,
            num_workers=config.n_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
        )

        if rank == 0: loop = tqdm(range(step, config.iteration), total=config.iteration, initial=step, desc='Train in progress')
        for epoch in range(10000):
            if ddp:
                data_loader_train.sampler.set_epoch(epoch)
            for batch in data_loader_train:
                step += 1
                if step > config.iteration:
                    break
                if rank ==0: loop.update(1)

                # train step
                trainer.model.train()
                # if epoch 
                loss, _ = train_step(trainer, batch, ema, optimizer, scheduler, scaler, clip_grad_norm, step, device, config)
                # logging train results (loss, lr)
                if rank == 0:
                    run.log({"train": dict(frame_loss=loss['loss'].mean(),
                                           kl=loss['kl'].mean(),
                                           decoder_nll=loss['decoder_nll'].mean(),
                                           loss1=loss['loss1'].mean(),
                                           loss2=loss['loss2'].mean(),
                                           )
                             }, step=step)
                    run.log({"lr": dict(lr=optimizer.state_dict()['param_groups'][0]['lr'])}, step=step)
                del loss, batch

                # TODO : sample with training model (if needed)
                # model.eval()
                # sample(batch, phase='train', step_type='iteration')
                # model.train()

                # validation step
                if step % config.valid_interval == 0 or step == 5000:
                    trainer.model.eval()
                    validation_metric = defaultdict(list)
                    with th.no_grad():
                        for n_valid, batch in enumerate(tqdm(data_loader_valid, desc='Validation in progress')):
                            batch_metric, _ = valid_step(trainer, batch, ema, step, device, config)
                            for k, v in batch_metric.items():
                                validation_metric[k].extend(v)
                            # for first batch, log image of feature map. shape(feature) = B C F L
                            if n_valid == 0 and rank == 0:
                                '''
                                # TODO: do this with hook
                                visual_range = 1000
                                for n in range(config.batch_size//world_size):
                                    fig, axes = plt.subplots(config.cnn_unit//6, 6, figsize=(8, 10))
                                    plt.axis('off')
                                    for m in range(config.cnn_unit):
                                        axes[m//6, m%6].imshow(feature[n,m].numpy()[:,:visual_range], aspect='auto', origin='lower')
                                    plt.subplots_adjust(wspace=0, hspace=0)
                                    run.log({'valid':{f'fmap_{n}': plt}}, step=step)
                                    plt.close()
                                '''

                    valid_mean = defaultdict(list)
                    if ddp:
                        output = [None for _ in range(world_size)]
                        dist.gather_object(validation_metric, output if rank==0 else None, dst=0)
                        if rank == 0:
                            for k,v in validation_metric.items():
                                if 'loss' in k:
                                    valid_mean[k] = th.mean(th.cat([th.stack(el[k]).cpu() for el in output]))
                                else:
                                    valid_mean[k] = np.mean(np.concatenate([el[k] for el in output]))
                    else:
                        for k,v in validation_metric.items():
                            if 'loss' in k:
                                valid_mean[k] = th.mean(th.stack(v).cpu())
                            else:
                                valid_mean[k] = np.mean(np.concatenate(v))
                    
                    if rank == 0:
                        print(f'validation metric: step:{step}')
                        run.log({'valid':valid_mean}, step=step)
                        for key, value in valid_mean.items():
                            if key[-2:] == 'f1' or 'loss' in key or key[-3:] == 'err':
                                print(f'{key} : {value}')
                        valid_mean['metric/note/f1']
                        model_saver.update(trainer, ema, optimizer, scheduler, step, valid_mean['metric/note/f1'], ddp=ddp)
                    if ddp:
                        dist.barrier()
            if step > config.iteration:
                break

    # Test phase
    trainer.model.eval()
    model_saver = ModelSaver(config, resume=True, order='higher')  # to load best model for all ranks
    best_ckp = model_saver.best_ckp
    # best_ckp = 'model_140k_0.9124.pt'
    print(colored(f"Best model is: {best_ckp}", "green", attrs=["bold"]))
    print(colored(f"Onset suppress for sampling: {config.diffusion_config['params']['onset_suppress_sample']}", "green", attrs=["bold"]))
    SAVE_PATH = config.logdir / (Path(best_ckp).stem + f'_eval_{config.dataset}_onsetsupp_{config.diffusion_config["params"]["onset_suppress_sample"]}_onsetweight_{config.diffusion_config["params"]["onset_weight_kl"]}')
    SAVE_PATH.mkdir(exist_ok=True)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ckp = th.load(model_saver.logdir / best_ckp, map_location=map_location)
    if ddp:
        trainer.model.module.load_state_dict(ckp['model_state_dict'])
        # TODO : load pretrain_model and ema
    else:
        trainer.model.load_state_dict(ckp['model_state_dict'])
        ema.load_state_dict(ckp['ema_state_dict'])
    
    test_set = get_dataset(config, ['test'], sample_len=None,
                            random_sample=False, transform=False)
    test_set.sort_by_length()
    batch_size = 8 # 6 for PAR model, 12G RAM (8 blocked by 8G shm size)
    if ddp:
        segments = np.split(np.arange(len(test_set)),
                            np.arange(len(test_set), step=batch_size))[1:]  # the first segment is []
        target_segments = [el for n, el in enumerate(segments) if n%world_size == rank]
        test_sampler = iter(target_segments)
    else:
        test_sampler = None
    if config.debug:
        test_sampler = iter([[0,1]])
    data_loader_test = DataLoader(
        test_set, batch_sampler=test_sampler,
        num_workers=config.n_workers,
        pin_memory=False,
        collate_fn=PadCollate()
        )
    test_metrics = defaultdict(list)

    iterator = data_loader_test
    with th.no_grad():
        for batch in tqdm(iterator, total=len(test_set), desc='Test in progress'):
            batch_metric, preds = test_step(trainer, batch, ema, step, device, config)
            for k, v in batch_metric.items():
                test_metrics[k].extend(v)
            for n in range(len(preds)):
                pred = preds[n].detach().cpu().numpy()
                np.savez(Path(SAVE_PATH) / (Path(batch['path'][n]).stem + '.npz'), pred=pred)

    test_mean = defaultdict(list)
    if ddp:
        output = [None for _ in range(world_size)]
        dist.gather_object(test_metrics, output if rank==0 else None, dst=0)
        if rank == 0:
            for k,v in test_metrics.items():
                if 'loss' in k:
                    test_mean[k] = th.cat([th.stack(el[k]).cpu() for el in output])
                else:
                    test_mean[k] = np.concatenate([el[k] for el in output])
    else:
        for k,v in test_metrics.items():
            if 'loss' in k:
                test_mean[k] = th.cat(th.stack(v).cpu())
            else:
                test_mean[k] = np.concatenate(v)
        
    if rank == 0:
        with open((Path(SAVE_PATH) / f'summary_{config.dataset}.txt'), 'w') as f:
            string, count = summary(model) 
            f.write(string + '\n')
            print('test metric')
            run.log({'test':test_mean}, step=step)
            for key, value in test_mean.items():
                if 'loss' not in key:
                    _, category, name = key.split('/')
                    multiplier = 100
                    if 'err' in key:
                        multiplier=1
                    metric_string = f'{category:>32} {name:26}: {np.mean(value)*multiplier:.3f} +- {np.std(value)*multiplier:.3f}'
                    print(metric_string)
                    f.write(metric_string + '\n')
                else:
                    metric_string = f'{key:>32}: {th.mean(value)*100:.3f} +- {th.std(value)*100:.3f}'
                    print(metric_string)
                    f.write(metric_string + '\n')
        wandb.finish()
    if ddp:
        dist.barrier()
        cleanup()
    
    
def run_demo(demo_fn, world_size, config):
    mp.spawn(demo_fn,
            args=(world_size, config),
            nprocs=world_size,
            join=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default='configs/VQ_Diffusion_S.json')
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-c', '--cnn_unit', type=int)
    parser.add_argument('-l', '--lstm_unit', type=int)
    parser.add_argument('-p', '--hidden_per_pitch', type=int)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume_dir', type=Path)
    parser.add_argument('--resume_id', type=str)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--no-ddp', dest='ddp', action='store_false')
    parser.add_argument('--finetune', action='store_true')

    # parser.add_argument('--amp', action='store_true', help='automatic mixture of precision')
    # parser.add_argument('--debug', action='store_true', help='set as debug mode')
    parser.set_defaults(ddp=True)
    parser.set_defaults(eval=False)
    
    args = parser.parse_args()
    config = default_config
    if args.config:
        with open(args.config, 'r') as j:
            update_config = json.load(j)
        print(update_config)
        config.update(update_config)
    for k, v in vars(args).items():
        if v is not None:
            config.update({k:v})
    config = SimpleNamespace(**config)
    if config.debug:
        config.valid_interval=20
        # config.valid_seq_len=160256
        config.iteration=20

    if args.resume_dir:
        id = args.resume_id
        config.id = id
        print(f'resume:{id}')
        config.logdir = args.resume_dir
    else:
        id = wandb.util.generate_id()
        config.id = id
        print(f'init:{id}')
        if hasattr(config, 'name'):
            config.logdir = Path('runs') / \
            ('_'.join([config.model, datetime.now().strftime('%y%m%d-%H%M%S'), config.name]))
        else:
            config.name=id
            config.logdir = Path('runs') / \
            ('_'.join([config.model, datetime.now().strftime('%y%m%d-%H%M%S'), id]))
        Path(config.logdir).mkdir(exist_ok=True)
    print(config)

    if not config.eval:
        dataset = get_dataset(config, ['train', 'validation', 'test'], random_sample=False, transform=False)
    else:
        os.environ['WANDB_DISABLED'] = 'true'
        dataset = get_dataset(config, None, random_sample=False, transform=False)
    dataset.initialize()

    
    if args.ddp:    
        n_gpus = th.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(train, world_size, config)
    else:
        train(rank=0, world_size=1, config=config, ddp=False)