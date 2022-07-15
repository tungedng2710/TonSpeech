import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlibrosa as tl

from src.CMGAN.models import generator
from src.CMGAN.utils import *

class CMGAN_ONNX(nn.Module):
    def __init__(self, 
                 checkpoint_path = None,
                 device_id = None,
                 n_fft = 400,
                 hop = 100):
        super(CMGAN_ONNX, self).__init__()
        if device_id == None:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:'+str(device_id))

        self.n_fft = n_fft
        self.cut_len = 16000*16
        self.hop = hop
        
        self.CMGAN = generator.TSCNet(num_channel=64, num_features=self.n_fft//2+1)
        self.CMGAN.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        self.CMGAN.to(self.device)
        self.CMGAN.eval()
        
        self.stft_extractor = tl.STFT(n_fft=n_fft, 
                                      hop_length=hop)
        self.istft_extractor = tl.ISTFT(n_fft=n_fft, 
                                        hop_length=hop)

    def forward(self, noisy):
        noisy = noisy.to(self.device)
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)
        length = noisy.size(-1)
        frame_num = int(np.ceil(length / 100))
        padded_len = frame_num * 100
        padding_len = padded_len - length
        noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
        if padded_len > self.cut_len:
            batch_size = int(np.ceil(padded_len/self.cut_len))
            while 100 % batch_size != 0:
                batch_size += 1
            noisy = torch.reshape(noisy, (batch_size, -1))
        stft = self.stft_extractor.forward(noisy.cpu())
        stft = torch.cat(stft, dim=0)
        stft = torch.movedim(stft, 0, 3)
        noisy_spec = torch.movedim(stft, 1, 2).to(self.device)

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag = self.CMGAN(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)

        real = est_spec_uncompress[:,:,:,0:1].movedim(3,0).movedim(3,2)
        imag = est_spec_uncompress[:,:,:,1:2].movedim(3,0).movedim(3,2)
        
        est_audio = self.istft_extractor.forward(real, imag, length=noisy.shape[-1])
        est_audio = est_audio / c
        with torch.no_grad():
            est_audio = torch.flatten(est_audio)[:length]
            assert len(est_audio) == length

        return est_audio


        



