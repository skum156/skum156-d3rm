import torch as th
from torch import nn
import numpy as np
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms
from natten import NeighborhoodAttention2D

# from .cqt import CQT
from transcription.constants import SR, HOP
# from .cqt import MultiCQT
from transcription.midispectrogram import CombinedSpec, MidiSpec

class NonARModel(nn.Module):
    def __init__(self, config, perceptual_w=False):
        super().__init__()
        self.model = config.model
        self.win_fw = config.win_fw
        self.win_bw = config.win_bw
        self.n_fft = config.n_fft
        self.hidden_per_pitch = config.hidden_per_pitch
        self.context_len = self.win_fw + self.win_bw + 1
        self.pitchwise = config.pitchwise_lstm
        if 'Mel2' in self.model:
            self.frontend = CombinedSpec()
        elif 'MIDI' in self.model:
            self.frontend = MIDIFrontEnd(config.n_per_pitch)
        else:
            self.frontend = transforms.MelSpectrogram(sample_rate=SR, n_fft=config.n_fft,
                hop_length=HOP, f_min=config.f_min, f_max=config.f_max, n_mels=config.n_mels, normalized=False)
        self.enhanced_context = config.enhanced_context
        if self.model == 'PAR_v2':
            self.acoustic = PAR_v2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film, cnn_widths=config.cnn_widths, multifc=config.multifc)
            self.vel_acoustic = PAR_v2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film, cnn_widths=config.cnn_widths, multifc=config.multifc)
        elif self.model == 'PAR_v2_MIDI_HPP':
            self.acoustic = PAR_v2_HPP(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film, cnn_widths=config.cnn_widths, n_per_pitch=config.n_per_pitch)
            self.vel_acoustic = PAR_v2_HPP(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film, cnn_widths=config.cnn_widths, n_per_pitch=config.n_per_pitch)
        self.lstm = nn.LSTM(config.hidden_per_pitch, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=True)
        self.output = nn.Linear(config.lstm_unit*2, 5)

        self.vel_lstm = nn.LSTM(config.hidden_per_pitch*2, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=True)
        self.vel_output = nn.Linear(config.lstm_unit*2, 1)

    def forward(self, audio, last_states=None, last_onset_time=None, last_onset_vel=None, 
                init_state=None, init_onset_time=None, init_onset_vel=None, sampling='gt', 
                max_step=400, random_condition=False, return_softmax=False):
        if sampling == 'gt':
            batch_size = audio.shape[0]
            conv_out, vel_conv_out = self.local_forward(audio)  # B x T x hidden x 88
            n_frame = conv_out.shape[1] 
            concat = conv_out.permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch)
            self.lstm.flatten_parameters()
            # print(f'concat:{concat.shape}')
            lstm_out, lstm_hidden = self.lstm(concat) # hidden_per_pitch
            # print(f'lstm:{lstm_out.shape}')
            frame_out = self.output(lstm_out) # n_frame, B*88 x n_class
            # print(f'frame_out:{frame_out.shape}')
            frame_out = frame_out.view(n_frame, batch_size, 88, 5).permute(1, 0, 2, 3) # B x n_frame x 88 x n_class
            vel_concat = th.cat((conv_out.detach(), vel_conv_out), dim=2).\
                permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch*2)
            self.vel_lstm.flatten_parameters()
            vel_lstm_out, vel_lstm_hidden = self.vel_lstm(vel_concat) # hidden_per_pitch
            vel_out = self.vel_output(vel_lstm_out) # n_frame, B*88 x 1
            vel_out = vel_out.view(n_frame, batch_size, 88).permute(1, 0, 2)  # B x n_frame x 88

            if return_softmax:
                frame_out = F.log_softmax(frame_out, dim=-1)

            return frame_out, vel_out

        else:
            batch_size = audio.shape[0]
            audio_len = audio.shape[1]
            step_len = (audio_len - 1) // HOP+ 1
            device = audio.device
 
            n_segs = ((step_len - 1)//max_step + 1)
            if 0 <= audio_len - (n_segs-1)*max_step* HOP< self.n_fft//2: # padding size of cqt
                n_segs -= 1
            seg_edges = [el*max_step for el in range(n_segs)]

            frame = th.zeros((batch_size, step_len, 88, 5)).to(device)
            vel = th.zeros((batch_size, step_len, 88)).to(device)

            offset = 0

            print(f'n_segs:{n_segs}, n_step:{step_len}')
            seg_num = 0
            for step in seg_edges:
                offset = step
                if step == 0:  # First segment
                    unpad_start = False
                    start = 0
                else:
                    del conv_out, vel_conv_out
                    unpad_start = True
                    start = offset * HOP - self.n_fft//2 

                if step == seg_edges[-1]:  # Last segment
                    unpad_end = False
                    end = None
                else:
                    # margin for CNN
                    end = (offset + max_step + 10) * HOP + self.n_fft//2
                    unpad_end = True
                
                conv_out, vel_conv_out = self.local_forward(
                    audio[:, start: end],
                    unpad_start=unpad_start, unpad_end=unpad_end)
                    
                n_frame = conv_out.shape[1] 
                concat = conv_out.permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch)
                self.lstm.flatten_parameters()
                # print(f'concat:{concat.shape}')
                lstm_out, lstm_hidden = self.lstm(concat) # hidden_per_pitch
                # print(f'lstm:{lstm_out.shape}')
                frame_out = self.output(lstm_out) # n_frame, B*88 x n_class
                # print(f'frame_out:{frame_out.shape}')
                frame_out = frame_out.view(n_frame, batch_size, 88, 5).permute(1, 0, 2, 3) # B x n_frame x 88 x n_class
                vel_concat = th.cat((conv_out.detach(), vel_conv_out), dim=2).\
                    permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch*2)
                self.vel_lstm.flatten_parameters()
                vel_lstm_out, vel_lstm_hidden = self.vel_lstm(vel_concat) # hidden_per_pitch
                vel_out = self.vel_output(vel_lstm_out) # n_frame, B*88 x 1
                vel_out = vel_out.view(n_frame, batch_size, 88).permute(1, 0, 2)  # B x n_frame x 88

                # frame_out[:, 0, :, [2,4]] *= 1.3  # increase the probability of 2 and 4
                frame[:, step:step+n_frame] = frame_out.squeeze(1)
                vel[:, step:step+n_frame] = vel_out.squeeze(1)

            if return_softmax:
                frame = F.log_softmax(frame, dim=-1)
            return frame, vel

    def local_forward(self, audio, unpad_start=False, unpad_end=False):
        if 'MIDI' in self.model:
            spec = self.frontend(audio)
            if unpad_start:
                spec = spec[:,:,:,self.n_fft//2//HOP:]
            if unpad_end:
                spec = spec[:,:,:,:-self.n_fft//2//HOP]
            conv_out = self.acoustic(spec)  # B x T x hidden_per_pitch x 88
            vel_conv_out = self.vel_acoustic(spec)
            return conv_out, vel_conv_out
        else:
            mel = self.frontend(
                audio[:, :-1]).transpose(-1, -2) # B L F
            mel = (th.log(th.clamp(mel, min=1e-9)) + 7) / 7
            if unpad_start:
                mel = mel[:,self.n_fft//2//HOP:]
            if unpad_end:
                mel = mel[:,:-self.n_fft//2//HOP]
            conv_out = self.acoustic(mel)  # B x T x hidden_per_pitch x 88
            vel_conv_out = self.vel_acoustic(mel) # B x T x hidden_per_pitch x 88

            return conv_out, vel_conv_out


class FilmLayer(nn.Module):
    def __init__(self, n_f, channel, hidden=16):
        super().__init__()
        pitch = (th.arange(n_f)/n_f).view(n_f, 1)
        self.register_buffer('pitch', pitch.float())
        self.alpha_linear = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel, bias=False),
        )
        self.beta_linear = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel, bias=False),
        )

    def forward(self, x):
        # x : shape of (B,C,L,F)
        alpha = self.alpha_linear(self.pitch).transpose(0,1).unsqueeze(1) # C, 1, F
        beta = self.beta_linear(self.pitch).transpose(0,1).unsqueeze(1) # C, 1, F
        x = alpha * x + beta
        return x


class FilmBlock(nn.Module):
    def __init__(self, n_input, n_unit, n_f, hidden=16, use_film=True, width_l1=3, width_l2=3):
        super().__init__()
        assert(width_l1 in [1,3])
        assert(width_l2 in [1,3])
        self.conv1 = nn.Conv2d(n_input, n_unit, (3, width_l1), padding='same')
        self.conv2 = nn.Conv2d(n_unit, n_unit, (3, width_l2), padding='same')
        self.bn = nn.BatchNorm2d(n_unit)
        self.use_film = use_film
        if use_film:
            self.film = FilmLayer(n_f, n_unit, hidden=hidden)

    def forward(self, x):
        if self.use_film:
            # x : shape of B C F L
            x = F.relu(self.conv1(x))
            x = self.film(x.transpose(2,3)).transpose(2,3)
            res = self.conv2(x)
            res = self.bn(res) 
            res = self.film(res.transpose(2,3)).transpose(2,3)
            x = F.relu(x + res)
            return x
        else:
            x = F.relu(self.conv1(x))
            res = self.conv2(x)
            res = self.bn(res) 
            x = F.relu(x + res)
            return x


class PAR(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = fc_x.transpose(1,2)  # B C L
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)).unsqueeze(3)
        fc_x = F.unfold(fc_x, (self.win_bw + self.win_fw + 1, 1)) 
        fc_x = self.win_fc(fc_x.transpose(1,2))  # B L C
        fc_x = fc_x.view(batch_size, -1, self.hidden_per_pitch, 88)

        x = self.layernorm(fc_x)
        return F.relu(x)
    
class PAR_v2(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], multifc=True):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[0], width_l2=cnn_widths[1]),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film, width_l1=cnn_widths[2], width_l2=cnn_widths[3]),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film, width_l1=cnn_widths[4], width_l2=cnn_widths[5]),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.multifc=multifc
        if multifc:
            self.win_fc = nn.Conv1d(fc_unit, fc_unit, self.win_fw+self.win_bw+1)
        self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        if self.multifc:
            fc_x = F.pad(fc_x.permute(0,2,1), (self.win_bw, self.win_fw)) # B C L 
            fc_x = self.win_fc(fc_x).transpose(1,2)
        pitchwise_x = self.pitch_linear(fc_x)
        pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88)
        return F.relu(self.layernorm(pitchwise_x))

class MIDIFrontEnd(nn.Module):
    def __init__(self, n_per_pitch=3, detune=0.0) -> None:
        # Detune: semitone unit. 0.5 means 50 cents.
        super().__init__()
        self.midi_low = MidiSpec(1024, n_per_pitch)
        self.midi_mid = MidiSpec(4096, n_per_pitch)
        self.midi_high = MidiSpec(8192, n_per_pitch)

    def forward(self, audio, detune_list=None):
        midi_low = self.midi_low(audio, detune_list)
        midi_mid = self.midi_mid(audio, detune_list)
        midi_high = self.midi_high(audio, detune_list)
        spec = th.stack([midi_low, midi_mid, midi_high], dim=1)
        return spec # B, 3, F, T

class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out, n_per_pitch=4, use_film=False, n_f=None) -> None:
        super().__init__()
        dilations = [round(12*np.log2(a)*n_per_pitch) for a in range(2, 10)]
        self.conv = nn.ModuleDict()
        for i, d in enumerate(dilations):
            self.conv[str(i)] = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, d])
        self.use_film = use_film
        if use_film:
            self.film = FilmLayer(n_f, c_out)
    def forward(self, x):
        x = self.conv['0'](x) + self.conv['1'](x) + self.conv['2'](x) + self.conv['3'](x) + \
            self.conv['4'](x) + self.conv['5'](x) + self.conv['6'](x) + self.conv['7'](x)
        if self.use_film:
            x = self.film(x)
        x = th.relu(x)
        return x

class PAR_v2_HPP(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
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

    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], n_per_pitch=5):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.block_1 = self.get_conv2d_block(3, cnn_unit, kernel_size=7)
        self.block_2 = self.get_conv2d_block(cnn_unit, cnn_unit, kernel_size=7)
        self.block_2_5 = self.get_conv2d_block(cnn_unit, cnn_unit, kernel_size=7)

        c3_out = 128
        
        self.conv_3 = HarmonicDilatedConv(cnn_unit, c3_out, n_per_pitch)
        self.conv_4 = HarmonicDilatedConv(c3_out, c3_out, n_per_pitch)
        self.conv_5 = HarmonicDilatedConv(c3_out, c3_out, n_per_pitch)

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, n_per_pitch], dilation=[1, 12*n_per_pitch])
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1])

    def forward(self, mel):
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


class ConvFilmBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, dilation, pool_size=None, use_film=True, n_f=88):
        super().__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation)
        self.relu = nn.ReLU()
        if use_film:
            self.film = FilmLayer(n_f, channel_out)
        if pool_size != None:
            self.pool = nn.MaxPool2d(pool_size)
        self.norm = nn.InstanceNorm2d(channel_out)
        self.pool_size = pool_size
        self.use_film = use_film

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.use_film:
            x = self.film(x)
        if self.pool_size != None:
            x = self.pool(x)
        x = self.norm(x)
        return x
