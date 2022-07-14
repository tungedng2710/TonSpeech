import numpy as np
import os
import torchaudio
import soundfile as sf
import argparse
import timeit

from src.CMGAN.models import generator
from src.CMGAN.tools.compute_metrics import compute_metrics
from src.CMGAN.utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--noisy", type=str, default="data/noisy_sample_16k.wav",
                    help="path/to/noisy/voice/(or the folder of noisy voice)")
parser.add_argument("--saved_folder", type=str, help="path/to/output/folder", default="data/CMGAN_exp")
parser.add_argument("--model_path", type=str, default='pretrained_models/CMGAN/cmgan_ckpt',
                    help="the path where the model is saved")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
args = parser.parse_args()

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cpu")

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    print("Loading noisy audio file...")
    noisy, sr = torchaudio.load(audio_path)
    print(audio_path + "has been loaded!")
    print("Enhancing...")
    assert sr == 16000
    noisy = noisy.to(DEVICE)

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len/cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).to(DEVICE), onesided=True)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).to(DEVICE),
                            onesided=True)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, "enhanced_"+name)
        sf.write(saved_path, est_audio, sr)
        print("Output file has been saved at ", saved_path)

if __name__ == "__main__":
    print("+-------------------------------+")
    print("|           TonSpeech           |")
    print("| Speech Enhancement with CMGAN |")
    print("+-------------------------------+")
    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder)
    start = timeit.default_timer()
    n_fft = 400
    print("Creating CMGAN Model...")
    model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).to(DEVICE)
    model.load_state_dict((torch.load(args.model_path)))
    model.eval()
    if os.path.isfile(args.noisy):
        enhance_one_track(model=model, 
                        audio_path=args.noisy, 
                        saved_dir=args.saved_folder, 
                        cut_len=16000*16, 
                        n_fft=n_fft, 
                        hop=n_fft//4, 
                        save_tracks=args.save_tracks)
    else:
        print("not supported")
    stop = timeit.default_timer()
    print("Done!")
    print('Inference time: {t}s'.format(t=round(stop-start, 2)))