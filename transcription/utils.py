# transcription/utils.py

import sys
import os
import torch
import torchaudio
from functools import reduce
from torch.nn.modules.module import _addindent
from torch.utils.data import Sampler
import numpy as np
import pretty_midi

def summary(model, file=sys.stdout):
    """Prints a summary of a PyTorch model with parameter count."""
    def repr_inner(module):
        extra_lines = []
        extra_repr = module.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        child_lines = []
        total_params = 0
        for key, child in module._modules.items():
            child_str, child_params = repr_inner(child)
            child_str = _addindent(child_str, 2)
            child_lines.append(f'({key}): {child_str}')
            total_params += child_params

        for name, p in module._parameters.items():
            if p is not None and hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        lines = extra_lines + child_lines
        main_str = module._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        if file is sys.stdout:
            main_str += f', \033[92m{total_params:,}\033[0m params'
        else:
            main_str += f', {total_params:,} params'
        return main_str, total_params

    string, count = repr_inner(model)
    print(string, file=file)
    return string, count


class CustomSampler(Sampler):
    """Sampler that iterates over a fixed list of indexes."""
    def __init__(self, indexes):
        self.indexes = indexes
        self.n_batch = len(indexes)

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.n_batch


def inference(audio_path, output_dir, model):
    import os
    import torch
    import torchaudio

    os.makedirs(output_dir, exist_ok=True)
    
    # Construct output MIDI file path.
    base_audio_filename = os.path.basename(audio_path)
    midi_filename = base_audio_filename.replace(".mp3", "").replace(".flac", "") + ".mid"
    output_midi_path = os.path.join(output_dir, midi_filename)

    print(f"Loading audio from: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True) # Convert to mono
    if sr != 16000:
        print(f"Resampling audio from {sr} Hz to 16000 Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.to(torch.float32)

    # Move waveform to model's device
    device = next(model.parameters()).device
    waveform = waveform.to(device)

    # Step 1: Encoder
    features_from_encoder = model.encoder(waveform.unsqueeze(0))
    print("features.shape after encoder (in utils):", features_from_encoder.shape)

    # Step 2: Dummy label and time step 't' tensor
    B = features_from_encoder.shape[0] # Batch size (1)
    
    # The SEQ_LEN for label needs to match the time dimension of the features
    SEQ_LEN_TIME = features_from_encoder.shape[-1] # This is 88 (from features [1,1,128,88])

    dummy_label_value = 0
    label = torch.full((B, SEQ_LEN_TIME), fill_value=dummy_label_value, dtype=torch.long, device=device)

    t = torch.full((B,), fill_value=999, dtype=torch.long, device=device)

    # Step 3: Forward pass through D3RM model
    prediction = model(features_from_encoder, label, t)
    print("Prediction shape from model:", prediction.shape)

    # --- Debug print to confirm execution flow ---
    print("--- Starting MIDI conversion process ---")

    # Convert prediction to CPU and numpy
    # prediction is [1, 88, 5] (Batch, TimeSteps, States)
    prediction_np = prediction.detach().cpu().numpy().squeeze(0) # [88, 5] (TimeSteps, States)

    # Get the predicted states by taking argmax across the 5 states for each time step
    predicted_states = np.argmax(prediction_np, axis=-1) # predicted_states is [88] (TimeSteps)

    # Define parameters for MIDI conversion
    sample_rate = 16000
    # You need to know the hop_length of your encoder's STFT/feature extraction.
    # Common hop_lengths are 512 or 256. If not specified, 512 is a reasonable guess.
    # If the encoder output 88 frames for a 1.76 second segment (88 * 512 / 16000),
    # then the frames per second is 88 / 1.76 = 50.
    # Let's calculate frames_per_second assuming a hop_length that gives 88 frames from a known audio duration.
    # Or, if this 88 frames is a fixed window, what real-world time does it represent?
    # Assuming standard hop_length=512 for 16kHz audio:
    hop_length = 512
    frames_per_second = sample_rate / hop_length # 16000 / 512 = 31.25

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # Map the 5 global states to distinct MIDI pitches (e.g., for visualization)
    midi_pitch_map = {
        0: 60, # C5
        1: 62, # D5
        2: 64, # E5
        3: 65, # F5
        4: 67, # G5
    }
    
    velocity = 100 # Fixed velocity for simplicity

    # Iterate through time steps (frames)
    for i in range(SEQ_LEN_TIME): # SEQ_LEN_TIME is 88
        state_idx = predicted_states[i] # Get the predicted state for this time step

        # Only add a note if the state is mapped and considered "active" (i.e., not necessarily state 0)
        # For this example, we add a note for ANY predicted state, just to see output.
        if state_idx in midi_pitch_map:
            pitch = midi_pitch_map[state_idx]
            start_time = i / frames_per_second
            end_time = (i + 1) / frames_per_second # Note lasts for one frame duration

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            piano.notes.append(note)

    midi.instruments.append(piano)

    # Save the MIDI file
    try:
        midi.write(output_midi_path)
        print(f"MIDI file successfully saved to: {output_midi_path}")
    except Exception as e:
        print(f"Error saving MIDI file: {e}")

    print("Inference process finished.") # Changed from "Inference completed successfully. Check output directory."