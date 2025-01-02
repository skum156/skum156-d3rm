import argparse
import os
from posixpath import basename
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
import pickle
from librosa.core import audio
from numpy.lib.npyio import save
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch as th
import torch.nn.functional as F
import mir_eval
import pretty_midi
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import match_note_offsets, precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
from torch.utils.data import DataLoader
import librosa

import transcription.dataset as dataset_module
from transcription.constants import HOP, MIN_MIDI, MAX_MIDI, SR
from transcription import *
from transcription import representation
from transcription.midi import save_midi
from transcription.decode import extract_notes
eps = sys.float_info.epsilon

est_note_attr = ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'time', 'length', 'length_long', 'pitch', 'velocity', 'onset_type', 'condition', 'exact_offset']
ref_note_attr = ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'time', 'length', 'length_long', 'pitch', 'velocity', 'onset_type', 'ioi', 'exact_offset'] 

def match_notes(
    tsv_path,
    pred,
    vel_pred=None,
    vel_label=None,
    save_path=None, 
    rep_type='base', 
    detail_eval=False, 
    save=False,
    tuning_factor=1,
    sequence_length=None,
    ):
    '''
    Inference audio of a sequence into piano roll, and save and
    evaluate it with original (not-quantized) midi.
    '''
    metrics = defaultdict(list)

    ######## Label ########
    # Parse midifile to get precise (unquantized) annotation.
    # onset_ref, offset_ref, frame_ref = representation.base2onsets_and_frames(label)

    '''
    if 'MAPS' in audio_path.parts:
        tsv_path = audio_path.parent.parent / 'tsv/matched' / audio_path.with_suffix('.tsv').name
    elif audio_path.parts[0] == '..':
        tsv_path = '/'.join(audio_path.parts[1:-1]) + '/' + audio_path.with_suffix('.tsv').name
    else:
        tsv_path = audio_path.with_suffix('.tsv')
    '''
    # tsv_path = audio_path.with_suffix('.tsv')
    midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

    onsets_ref, offsets_ref, pitches_ref, vels_ref = zip(*midi)
    offset_cut = []
    reonset = [] # 0 False onset, True denote re-onset
    for n in range(len(onsets_ref)):
        m = n+1
        offset = offsets_ref[n]
        while m < len(onsets_ref) and offsets_ref[n] > onsets_ref[m]:
            # cut offset when another onset appears
            if pitches_ref[m] == pitches_ref[n]:
                offset = onsets_ref[m]
                reonset.append(True) 
                break
            m += 1
        else:
            reonset.append(False)
                
        offset_cut.append(offset)
    reonset = np.asarray(reonset)
    pitches_ref = np.asarray(pitches_ref)
    onsets_ref = np.asarray(onsets_ref)

    # calulate inter onset interval
    intervals = np.zeros((len(onsets_ref),))
    for pitch in range(MIN_MIDI, MAX_MIDI+1):
        idx = np.flatnonzero(pitches_ref==pitch)
        if len(idx) != 0:
            onset_subset = onsets_ref[idx]
            ioi = np.diff(onset_subset)
            intervals[idx[1:]] = ioi
    
    # p_ref = np.array([midi_to_hz(pitch) for pitch in pitches_ref])
    p_ref = pitches_ref
    f_ref = np.array([midi_to_hz(pitch) for pitch in pitches_ref])
    i_ref = np.array(list(zip(onsets_ref, offset_cut)))
    i_ref_long = np.array(list(zip(onsets_ref, offsets_ref)))
    v_ref = np.array(vels_ref)


    ######## Prediction ########
    try:
        velocities = np.clip(vel_pred*128, 0, 128)
    except:
        velocities = vel_label 
    onset_est = ((pred == 2) + (pred == 4))
    frame_est = ((pred == 2) + (pred == 3) + (pred == 4))
    p_est, i_est, v_est = extract_notes(onset_est, frame_est, velocities)

    # p_est = MIN_MIDI + p_est
    scaling = HOP/ SR
    i_est = i_est * scaling
    f_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    if tuning_factor != 1:
        i_est = (i_est * tuning_factor).reshape(-1, 2)

    '''
    ######### Frame ########
    t_est, f_est = notes_to_frames(p_est, i_est, frames.shape)

    scaling = HOP_LENGTH / SAMPLE_RATE 
    i_est = (i_est * scaling).reshape(-1, 2)
    if tuning_factor != 1:
        i_est /= tuning_factor
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_est = t_est.astype(np.float64) * scaling
    if tuning_factor != 1:
        t_est /= tuning_factor
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                        for midi in freqs]) for freqs in f_est]
    # if tuning:
    #     t_est /= detune_ratio

    # frame_metrics = evaluate_frames(t_ref, p_ref, t_est, f_est)
    metrics['metric/frame/f1'].append(hmean(
        [frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    for key, value in frame_metrics.items():
        metrics['metric/frame/' + key.lower().replace(' ', '_')].append(value)
    '''

    if sequence_length:
        idx = np.where(i_ref[:,0] < sequence_length)
        i_ref = np.clip(i_ref[idx], 0, sequence_length)
        i_ref_long = np.clip(i_ref_long[idx], 0, sequence_length)
        f_ref = f_ref[idx]
        p_ref = p_ref[idx]
        v_ref = v_ref[idx]
        intervals = intervals[idx]
        reonset = reonset[idx]
    
    ######## Matching ########

    # Est_note = namedtuple('Est_note', ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'length', 'pitch', 'velocity', 'onset_type', 'condition', 'exact_offset']) 
    # Ref_note = namedtuple('Ref_note', ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'length', 'pitch', 'velocity', 'onset_type', 'ioi', 'exact_offset']) 
    matching = mir_eval.transcription.match_notes(i_ref, f_ref, i_est, f_est, offset_ratio=None)
    matching_off = mir_eval.transcription.match_notes(i_ref, f_ref, i_est, f_est)
    matching_off_exact = mir_eval.transcription.match_notes(i_ref, f_ref, i_est, f_est, offset_ratio=0.0)
    match_notes_velocity = mir_eval.transcription_velocity.match_notes(i_ref, f_ref, v_ref, i_est, f_est, v_est, offset_ratio=None)
    
    ref_notes = np.zeros((len(i_ref), len(ref_note_attr)))
    est_notes = np.zeros((len(i_est), len(est_note_attr)))

    ref_notes[:, ref_note_attr.index('pitch')] = p_ref
    ref_notes[:, ref_note_attr.index('velocity')] = v_ref
    ref_notes[:, ref_note_attr.index('time')] = i_ref[:,0]
    ref_notes[:, ref_note_attr.index('length')] = np.diff(i_ref, axis=1).squeeze()
    ref_notes[:, ref_note_attr.index('length_long')] = np.diff(i_ref_long, axis=1).squeeze()
    ref_notes[:, ref_note_attr.index('onset_type')] = np.asarray(reonset)
    ref_notes[:, ref_note_attr.index('ioi')] = intervals

    est_notes[:, est_note_attr.index('pitch')] = p_est
    est_notes[:, est_note_attr.index('velocity')] = v_est
    est_notes[:, est_note_attr.index('length')] = np.diff(i_est, axis=1).squeeze()
    est_notes[:, est_note_attr.index('time')] = i_est[:,0]
    ref_idx, est_idx = zip(*matching)
    ref_notes[ref_idx, ref_note_attr.index('match')] = 1
    est_notes[est_idx, est_note_attr.index('match')] = 1

    ref_idx, est_idx = zip(*matching_off)
    ref_notes[ref_idx, ref_note_attr.index('offset_match')] = 1
    est_notes[est_idx, est_note_attr.index('offset_match')] = 1

    ref_idx, est_idx = zip(*matching_off_exact)
    ref_notes[ref_idx, ref_note_attr.index('exact_offset')] = 1
    est_notes[est_idx, est_note_attr.index('exact_offset')] = 1

    ref_idx, est_idx = zip(*match_notes_velocity)
    ref_notes[ref_idx, ref_note_attr.index('velocity_match')] = 1
    est_notes[est_idx, est_note_attr.index('velocity_match')] = 1


    # what to calculate?
    # acc per note Length
    # precise offset eval
    # how reonset was evaluated
    # how acc of (onset, condition) differ

    return ref_notes, est_notes, matching


def standard_metric(ref_notes, est_notes, matching):
    def calculate_recall_precision_f1(ref_notes, est_notes, attr):
        recall = sum(ref_notes[:, ref_note_attr.index(attr)]) / len(ref_notes)
        precision = sum(est_notes[:, est_note_attr.index(attr)]) / len(est_notes)
        f1 = 2*recall*precision /(recall + precision)
        return recall, precision, f1

    metrics = defaultdict(list)

    if matching:
        ref_idx, est_idx = zip(*matching)
        v_ref_match = ref_notes[np.array(ref_idx), ref_note_attr.index('velocity')]
        v_est_match = est_notes[np.array(est_idx), est_note_attr.index('velocity')]

        err = v_ref_match - v_est_match
        metrics['velocity/abs_err'].append(np.mean(np.abs(err)))
        metrics['velocity/rel_err'].append(np.mean(np.abs(err) / v_ref_match))

    onset_recall, onset_precision, onset_f1 = calculate_recall_precision_f1(ref_notes, est_notes, 'match')
    metrics['onset/recall'].append(onset_recall)
    metrics['onset/precision'].append(onset_precision)
    metrics['onset/f1'].append(onset_f1)

    offset_recall, offset_precision, offset_f1 = calculate_recall_precision_f1(ref_notes, est_notes, 'offset_match')
    metrics['offset/recall'].append(offset_recall)
    metrics['offset/precision'].append(offset_precision)
    metrics['offset/f1'].append(offset_f1)

    offset_recall_l, offset_precision_l, offset_f1_l = calculate_recall_precision_f1(ref_notes, est_notes, 'offset_match_long')
    metrics['offset_long/recall'].append(offset_recall_l)
    metrics['offset_long/precision'].append(offset_precision_l)
    metrics['offset_long/f1'].append(offset_f1_l)

    idx = np.where(ref_notes[:, ref_note_attr.index('match')] == 1)
    n_match = np.sum(ref_notes[idx, ref_note_attr.index('velocity_match')])
    onset_vel_recall = n_match / len(ref_notes)
    idx = np.where(est_notes[:, est_note_attr.index('match')] == 1)
    n_match = np.sum(est_notes[idx, est_note_attr.index('velocity_match')])
    onset_vel_precision = n_match / len(est_notes)
    onset_vel_f1 = 2*onset_vel_recall*onset_vel_precision /(onset_vel_recall + onset_vel_precision)
    metrics['onset_w_vel/recall'].append(onset_vel_recall)
    metrics['onset_w_vel/precision'].append(onset_vel_precision)
    metrics['onset_w_vel/f1'].append(onset_vel_f1)

    idx = np.where(ref_notes[:, ref_note_attr.index('offset_match')] == 1)
    n_match = np.sum(ref_notes[idx, ref_note_attr.index('velocity_match')])
    offset_vel_recall = n_match / len(ref_notes)
    idx = np.where(est_notes[:, est_note_attr.index('offset_match')] == 1)
    n_match = np.sum(est_notes[idx, est_note_attr.index('velocity_match')])
    offset_vel_precision = n_match / len(est_notes)
    offset_vel_f1 = 2*offset_vel_recall*offset_vel_precision /(offset_vel_recall + offset_vel_precision)
    metrics['offset_w_vel/recall'].append(offset_vel_recall)
    metrics['offset_w_vel/precision'].append(offset_vel_precision)
    metrics['offset_w_vel/f1'].append(offset_vel_f1)

    return metrics



def infer_dataset(model_file, dataset, dataset_group, sequence_length, save_path, rep_type, n_class, no_recursive,
                  onset_threshold, frame_threshold, device='cpu', gt_condition=False, offset_bias=2, delay=1, cal_loss=False, tuning=False):
    if save_path == None:
      save_path = os.path.dirname(model_file)
    Path(save_path).mkdir(exist_ok=True)
    recursive = not no_recursive
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length, 'delay': delay}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model_state_path = model_file
    ckp = th.load(model_state_path, map_location='cpu')
    model_name = ckp['model_name']

    # model_class = getattr(models, model_name)
    if model_name == 'Autoregressive':
        model = models.AutoRegressiveModel(
            ckp['n_mels'],
            ckp['n_fft'],
            ckp['f_min'],
            ckp['f_max'],
            ckp['cnn_unit'],
            ckp['lstm_unit'],
            ckp['fc_unit'],
            ckp['hidden_per_pitch'],
            ckp['d_feedforward'],
            ckp['d_context'],
            24,
            ckp['n_class'],
            ckp['win_fw'],
            ckp['win_bw'],
            ckp['n_head'],
            ckp['ac_model_type'],
            ckp['vel_model_type'],
        )
    elif model_name == 'Autoregressive_v2':
        model = models.AutoRegressiveModel_v2(
            ckp['n_mels'],
            ckp['n_fft'],
            ckp['f_min'],
            ckp['f_max'],
            ckp['cnn_unit'],
            ckp['lstm_unit'],
            ckp['fc_unit'],
            ckp['hidden_per_pitch'],
            ckp['n_class'],
            ckp['win_fw'],
            ckp['win_bw'],
            ckp['ac_model_type'],
            ckp['vel_model_type'],
        )
    elif model_name == 'Autoregressive_v0':
        model = models.AutoRegressiveModel_v0(
            ckp['n_mels'],
            ckp['n_fft'],
            ckp['f_min'],
            ckp['f_max'],
            ckp['cnn_unit'],
            ckp['lstm_unit'],
            ckp['fc_unit'],
            ckp['hidden_per_pitch'],
            ckp['n_class'],
            ckp['win_fw'],
            ckp['win_bw'],
            ckp['ac_model_type'],
            ckp['vel_model_type'],
        )
    elif model_name == 'onsets_and_frames':
        model = onsets_and_frames.OnsetsAndFrames(
            ckp['n_mels'], 
            ckp['n_fft'],
            ckp['f_min'],
            ckp['f_max'],
            ckp['cnn_unit'],
            ckp['fc_unit'],
            bidirectional=True)
    elif model_name == 'FlexibleModel':
        model = models.FlexibleModel(
            ckp['n_mels'],
            ckp['n_fft'],
            ckp['f_min'],
            ckp['f_max'],
            ckp['cnn_unit'],
            ckp['lstm_unit'],
            ckp['fc_unit'],
            False, True,
            ckp['ac_model_type'],
            ckp['lm_model_type'],
            ckp['n_class'],
            ckp['win_fw'],
            ckp['win_bw'],
        )


    model.load_state_dict(ckp['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    n_bin = 20
    cum_acc_conf = np.zeros((n_class, n_bin, 3))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    cum_metrics = defaultdict(list)
    sampling_method = 'gt' if gt_condition else 'argmax'
    # from .core.utils import cycle
    '''
    model.share_memory()
    num_processes = 8
    work = partial(
        process_sample, 
        save_path=save_path, 
        model=model, 
        device=device, 
        model_name=model_name, 
        rep_type=rep_type,
        recursive=recursive,
        delay=delay,
        cal_loss=cal_loss,
        tuning=tuning)
    '''
    for batch in tqdm(loader):
        '''
        shifted_label = batch['shifted_label'].cuda()[:, 1:]
        pesudo_shape = list(shifted_label.shape)
        pesudo_shape.insert(1, 5)
        pesudo_train = torch.zeros(tuple(pesudo_shape)).cuda()
        try:
            criterion = NLLLoss()
            criterion(pesudo_train, shifted_label)
        except:
            print(basename)
        basename = Path(save_path) / Path(batch['path'][0]).stem
        save_name = str(basename) + f'_pred.npy'
        if torch.any(0 > shifted_label) or torch.any(shifted_label > 5):
            print(basename)
        if Path(save_name).exists:
            continue
        '''
        basename = Path(save_path) / Path(batch['path'][0]).stem
        save_name = str(basename) + f'_score.dat'
        if Path(save_name).exists():
            with open(save_name, 'rb') as f:
                metrics = pickle.load(f)
                print(metrics)
        else:
            metrics, acc_conf = match_notes(batch, model, device, model_name=model_name, save_path=save_path, criterion=NLLLoss(), plot_example=False, rep_type=rep_type,
                                        recursive=recursive, detail_eval=False, delay=delay, save=True, cal_loss=cal_loss, tuning=tuning)
        for key, value in metrics.items():
            cum_metrics[key].append(value)
            
        # cum_acc_conf += acc_conf
    '''
    ece = np.abs(cum_acc_conf[:, :, 0] - cum_acc_conf[:, :, 1]) * cum_acc_conf[:, :, 2]
    ece = np.nansum(ece, axis=-1) / np.nansum(cum_acc_conf[:, :, 2], axis=-1)
    np.save(Path(save_path) / f'acc_conf_{sampling_method}.npy', acc_conf)
    np.save(Path(save_path) / f'ece_{sampling_method}.npy', ece)
    '''

    with open(os.path.join(save_path, f'summary.txt'), 'w') as f:
      for key, values in cum_metrics.items():
          if key.startswith('metric/'):
              _, category, name = key.split('/')
              metric_string = f'{category:>32} {name:26}: {np.mean(values):.3f} +- {np.std(values):.3f}'
              print(metric_string)
              f.write(metric_string + '\n')

def process_sample(sample, save_path, model, device, model_name, rep_type, recursive, delay, cal_loss, tuning):
    basename = Path(save_path) / Path(sample['path'][0]).stem
    save_name = str(basename) + f'_score.dat'
    if Path(save_name).exists():
        with open(save_name, 'rb') as f:
            metrics = pickle.load(f)
            print(metrics)
    else:
        metrics, acc_conf = match_notes(sample, model, [device], model_name=model_name, save_path=save_path, criterion=NLLLoss(), plot_example=False, rep_type=rep_type,
                                    recursive=recursive, detail_eval=False, delay=delay, save=True, cal_loss=cal_loss, tuning=tuning)
    return metrics

def framewise_eval(argmax_pred, label):
    '''
    evaluate frame-wise (point-wise) evaluation
    argmax_pred: torch.tensor shape of (frame, pitch)
    label: torch.tensor shape of (frame, pitch)
    '''
    frame_metrics = defaultdict(list)

    n_class = label.max() - label.min() + 1
    for n in range(int(n_class)):
        tp = th.sum((label == n) * (argmax_pred == n))
        fn = th.sum((label == n) * (argmax_pred != n))
        fp = th.sum((label != n) * (argmax_pred == n))
        
        pr = tp / float(tp + fp)
        re = tp / float(tp + fn)
        f1 = 2 * pr * re / float(pr + re)
       
        frame_metrics[f'class_{n}/precision'] = pr
        frame_metrics[f'class_{n}/recall'] = re
        frame_metrics[f'class_{n}/f1'] = f1
    
    frame_metrics['accuracy'] = th.sum(argmax_pred == label) / float(label.numel())
    return frame_metrics

    
def extract_notes_with_reonset(
    pred,
    velocity=None, 
    onset_threshold=0.5, 
    frame_threshold=0.5, 
    defalut_velocity=64):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    reonsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets, offsets, frames = representation.convert2onsets_and_frames(pred, 'base')
    onsets = onsets.astype(int)
    offsets = offsets.astype(int)
    frames = frames.astype(int)
    onset_diff = np.concatenate(
        [onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

    if velocity is None:
        velocity = np.ones_like(onsets) * defalut_velocity

    pitches = []
    intervals = []
    intervals_long = []
    velocities = []
    conditions = []
    reonsets = []
    onset_frame, onset_pitch = onset_diff.nonzero()
    for pitch, frame in zip(onset_pitch, onset_frame):
        # frame = nonzero[0].item()
        # pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []
        offset_short = None

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break
            if (offset != onset) and onsets[offset, pitch].item():
                if offset_short == None:
                    offset_short = offset
        if offset_short == None:
            offset_short = offset

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset_short])
            intervals_long.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
            if frame == 0:
                conditions.append(0)
            else:
                conditions.append(pred[frame-1, pitch])
            if pred[frame, pitch] == 4:
                reonsets.append(True)
            else:
                reonsets.append(False)
        else:
            raise Exception

    return np.array(pitches), np.array(intervals), np.array(intervals_long), np.array(velocities), np.array(conditions), np.array(reonsets)


def my_func(input_tuple):
    if len(input_tuple) == 4:
        return match_notes(input_tuple[0], input_tuple[1], input_tuple[2], save_path=input_tuple[3])
    else:
        return match_notes(input_tuple[0], input_tuple[1], input_tuple[2], save_path=input_tuple[3], sequence_length=input_tuple[4])


'''
def preprocess():
    acc_conf = []
    
    assert batch['audio'].shape[0] == 1

    with th.no_grad():
        if 'onsets_and_frames' in model_name:
            frame_preds, onset_preds, offset_preds, vel_preds, losses, vel_losses = onsets_and_frames.run_on_batch(model, batch, device, delay=delay)
        elif 'Autoregressive' in model_name:
            preds, vel_preds, losses, vel_losses = models.run_on_batch(
                model, batch, device, sampling='argmax', criterion=criterion, rep_type=rep_type, delay=delay, cal_loss=cal_loss)
        else:
            preds, vel_preds, losses, vel_losses = models.run_on_batch(
                model, batch, device, sampling='argmax', criterion=criterion, rep_type=rep_type, delay=delay, cal_loss=cal_loss)
    if cal_loss:
        losses = losses.cpu().numpy()
        metrics['loss/frame_loss'].extend(list(np.atleast_1d(losses)))
        vel_losses = vel_losses.cpu().numpy()
        metrics['loss/velocity_loss'].extend(list(np.atleast_1d(vel_losses)))

    label = dict()
    for key in batch:
        label[key] = batch[key][0]
    
    if 'Autoregressive' in model_name:
        pred = preds[0]
        argmax_pred = pred.argmax(dim=-1)
        if detail_eval:
            acc_conf.append(calculate_acc_conf(pred.cpu().numpy().transpose((1, 2, 0)),
                                        label['shifted_label'][delay:].cpu().numpy()))
        else:
            acc_conf.append(None)
        onsets, offsets, frames = representation.convert2onsets_and_frames(argmax_pred, rep_type)
    elif 'FlexibleModel' in model_name:
        pred = preds[0]
        argmax_pred = pred.argmax(dim=-1)
        if detail_eval:
            acc_conf.append(calculate_acc_conf(pred.cpu().numpy().transpose((1, 2, 0)),
                                        label['shifted_label'][delay:].cpu().numpy()))
        else:
            acc_conf.append(None)
        onsets, offsets, frames = representation.convert2onsets_and_frames(argmax_pred, rep_type)
    elif model_name == 'onsets_and_frames':
        onsets = onset_preds[0] > 0.5
        offsets = offset_preds[0] > 0.5
        frames = frame_preds[0] > 0.5
        pred = frames
        pred[onsets] = 2
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str)
    parser.add_argument('dataset', nargs='?', default='MAESTRO_V2')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--postfix', default='')
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--n_class', default=5, type=int)
    parser.add_argument('--no_recursive', action='store_true')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--gt_condition', action='store_true')
    parser.add_argument('--offset_bias', default=1, type=float)
    parser.add_argument('--delay', default=1, type=int)
    parser.add_argument('--cal_loss', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--merge_onset_prob', action='store_true')
    parser.add_argument('--prob_weight', nargs=5, default=None, type=int)

    args = parser.parse_args()


    dataset_class = getattr(dataset_module, args.dataset)
    kwargs = {'sequence_length': args.sequence_length}
    if args.dataset_group is not None:
        kwargs['groups'] = [args.dataset_group]
    dataset = dataset_class(**kwargs)
    paths = [Path(el[0]) for el in dataset.data_path]
    tsv_paths = [Path(el[1]) for el in dataset.data_path]

    pickle_path = Path(args.save_path) / f'eval_notes{args.postfix}.pkl'
    if False:
    # if pickle_path.exists():
        eval_notes = pickle.load(open(pickle_path, 'rb'))
        # raise Exception(FileExistsError)
    else:
        pred = []
        vel_pred = []
        pred_paths = []
        for path in paths:
            save_data = Path(args.save_path) / path.name.replace('.flac', '.npz')
            dat = np.load(save_data)
            pred_paths.append(path)
            print(dat['pred'].shape)
            pred.append(th.from_numpy(dat['pred']))
            vel_pred.append(th.from_numpy(dat['pred']))
        if args.sequence_length:
            with Pool(processes=8) as pool:
                eval_notes = list(tqdm(
                    pool.imap(
                        my_func, 
                        zip(tsv_paths, pred, vel_pred, pred_paths, [args.sequence_length/16000 for _ in range(len(paths))])
                        ),
                    total=len(dataset)
                ))
        else:
            with Pool(processes=8) as pool:
                eval_notes = list(tqdm(
                    pool.imap(
                        my_func, 
                        zip(tsv_paths, pred, vel_pred, pred_paths)
                        ),
                    total=len(dataset)
                ))

        with open(pickle_path, 'wb') as f:
            pickle.dump(eval_notes, f)

    piecewise_metrics = defaultdict(list)

    total_ref = []
    total_est = []
    for ref_notes, est_notes, matching in eval_notes:
        # ref_notes, est_notes, matching = note_tuple
        metrics = standard_metric(ref_notes, est_notes, matching)
        for key, value in metrics.items():
            piecewise_metrics[key].append(value)
        total_ref.append(ref_notes)
        total_est.append(est_notes)
    total_ref = np.concatenate(total_ref)
    total_est = np.concatenate(total_est)
    total_metrics = standard_metric(total_ref, total_est, None)
    
    with open(os.path.join(args.save_path, f'summary{args.postfix}.txt'), 'w') as f:
        f.write('Piecewise Metrics \n')
        print('Piecewise Metrics \n')
        for key, values in piecewise_metrics.items():
            category, name = key.split('/')
            metric_string = f'{category:>32} {name:26}: {np.mean(values)*100:.3f} +- {np.std(values)*100:.3f}'
            print(metric_string)
            f.write(metric_string + '\n')

        f.write('Macro Metrics \n')
        print('Macro Metrics \n')
        for key, values in total_metrics.items():
            category, name = key.split('/')
            metric_string = f'{category:>32} {name:26}: {np.mean(values)*100:.3f} +- {np.std(values)*100:.3f}'
            print(metric_string)
            f.write(metric_string + '\n')
