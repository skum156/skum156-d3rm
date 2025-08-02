import argparse
import os
import torch
from transcription.module import D3RM
from transcription.encoder import ARModel
from transcription.decoder import Decoder
from transcription.loss import FocalLoss
from transcription.utils import inference

def main():
    parser = argparse.ArgumentParser(description='D3RM Inference (Simplified)')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('--ckpt_path', type=str, default='model_170k_0.9063_nonar.pt')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    use_vel = False
    label_embed_dim = 32
    lstm_dim = 128
    n_layers = 2
    window = 25
    dilation = [1, 1]
    condition_method = 'none'
    diffusion_step = 1000
    timestep_type = 'continuous'
    natten_direction = 'none'
    spatial_size = [88, 1000]
    num_state = 5
    classifier_free_guidance = False
    loss_alpha = 0.25
    loss_gamma = 2.0
    learning_rate = 1e-4

    encoder = ARModel(use_vel=use_vel)

    decoder_parameters = {
        'label_embed_dim': label_embed_dim,
        'lstm_dim': lstm_dim,
        'n_layers': n_layers,
        'window': window,
        'dilation': dilation,
        'condition_method': condition_method,
        'diffusion_step': diffusion_step,
        'timestep_type': timestep_type,
        'natten_direction': natten_direction,
        'spatial_size': spatial_size,
        'num_state': num_state,
        'classifier_free_guidance': classifier_free_guidance,
    }
    decoder = Decoder(**decoder_parameters)

    loss_fn = FocalLoss(alpha=loss_alpha, gamma=loss_gamma, reduction='mean')

    model = D3RM(
        encoder=encoder,
        decoder=decoder,
        loss_fn=loss_fn,
        lr=learning_rate,
    )

    try:
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
        model_state_dict = None
        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint

        if model_state_dict is None:
            raise ValueError("Could not find a valid model state_dict within the checkpoint.")

        new_state_dict = {}
        if model_state_dict and next(iter(model_state_dict), '').startswith('model.'):
            for k, v in model_state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
        else:
            new_state_dict = model_state_dict

        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(new_state_dict, strict=False)

    except Exception as e:
        print(f"An error occurred during checkpoint loading: {e}")
        return

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inference(args.input, args.output, model)

if __name__ == '__main__':
    main()
