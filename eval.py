from scipy.io import wavfile
from pesq import pesq   
from pystoi import stoi
import argparse 

import torch
import torchaudio
from torchaudio.transforms import Resample

def get_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--down_sample', type=int, help='down sample rate into 16k Hz')
    parser.add_argument('--clean', type=str, help='path/to/clean/voice')
    parser.add_argument('--denoised', type=str, help='path/to/denoised/voice')
    parser.add_argument('--metric', type=str, help='pesq or stoi', default='pesq')
    return parser.parse_args()

def load_sample(path: str = None,
                down_sample: bool = False):
    """
    path: path/to/your/audio/file
    """
    assert path is not None

    signal, rate = torchaudio.load(path)

    if down_sample:
        print("Changing sample rate to 16k Hz...")
        downsampler = Resample(orig_freq=rate, new_freq=16000)
        downsampled_signal = downsampler(signal.view(1, -1))
        signal = downsampled_signal
        rate = 16000
        print("Sample rate changed!")
    else:
        print("Warning: sample rate = ", rate)

    signal = torch.flatten(signal).numpy()
    return signal, rate

def eval_pesq(fs, clean, denoised):
    '''
    clean: ideal audio file
    denoised: noisy voice after performing speech enhancement
    '''
    ref = clean
    deg = denoised
    rate = fs
    if rate == 16000:
        return pesq(rate, ref, deg, 'wb')
    elif rate == 8000:
        return pesq(rate, ref, deg, 'nb')
    else:
        print("|  Please change sample rating into 16k or 8k")
        return "N/A"

def eval_stoi(fs, clean, denoised):
    '''
    clean: ideal audio file
    denoised: noisy voice after performing speech enhancement
    '''
    return stoi(clean, denoised, fs, extended=False)

def run_eval():
    # Get the test sample
    arg = get_arg()
    if arg.down_sample==1:
        down_sample = True
    else:
        down_sample = False
    clean, fs = load_sample(path=arg.clean, down_sample=down_sample)
    denoised, fs = load_sample(arg.denoised, down_sample=arg.down_sample)
    min_length = min(clean.shape[0], denoised.shape[0])
    clean = clean[:min_length]
    denoised = denoised[:min_length]

    # Print the result to console
    print("-----------------------------------------------------------------")
    print("|  Clean voice: ", arg.clean)
    print("|  Denoised voice: ", arg.denoised)
    if arg.metric == 'pesq':
        print("|  PESQ score: ", eval_pesq(fs, clean, denoised))
    elif arg.metric == 'stoi':
        print("|  STOI score: ", eval_stoi(fs, clean, denoised))
    else:
        print("|  The given metric isn't supported!")
    print("-----------------------------------------------------------------")

if __name__ == '__main__':
    run_eval()