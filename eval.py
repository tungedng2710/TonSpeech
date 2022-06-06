from pesq import pesq   
from pystoi import stoi
import argparse 
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.transforms import Resample

def get_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--down_sample', type=int, help='down sample rate into 16k Hz', default=1)
    parser.add_argument('--clean', type=str, help='path/to/clean/voice')
    parser.add_argument('--denoised', type=str, help='path/to/denoised/voice')
    parser.add_argument('--metric', type=str, help='pesq or stoi', default='pesq')
    parser.add_argument('--trimmed_duration', type=int, default=-1)
    return parser.parse_args()

def load_sample(path: str = None,
                down_sample: bool = False):
    """
    path: path/to/your/audio/file
    """
    assert path is not None
    signal, rate = torchaudio.load(path)

    if down_sample:
        downsampler = Resample(orig_freq=rate, new_freq=16000)
        downsampled_signal = downsampler(signal.view(1, -1))
        signal = downsampled_signal
        rate = 16000
        # print("Sample rate changed into 16kHz!")
    else:
        print("Warning: sample rate = ", rate)

    signal = torch.flatten(signal).numpy()
    return signal, rate

def get_batches(data, rate, duration):
    total_duration = len(data)/rate
    if duration >= total_duration:
        duration = total_duration
    batch_size = rate * duration
    batches = []
    remain_length = len(data)
    for i in range(0, len(data), batch_size):
        remain_length = remain_length - batch_size
        if remain_length >= batch_size:
            batches.append(data[i:i+batch_size])
        else:
            batches.append(data[i:])
            break
    return batches

def eval_pesq(fs: int = 16000, 
              clean = None, 
              denoised = None,
              trimmed_duration = -1):
    '''
    fs: sample rate
    clean: ideal audio file
    denoised: noisy voice after performing speech enhancement
    trimmed_duration: max duration for each batch 
    '''
    ref = clean
    deg = denoised
    rate = fs
    if trimmed_duration != -1:
        scores = []
        refs = get_batches(ref, rate, trimmed_duration)
        degs = get_batches(deg, rate, trimmed_duration)
        for i in tqdm(range(len(refs))):
            try:
                scores.append(pesq(rate, refs[i], degs[i], 'wb'))
                # print("Batch %d: pesq score: %f", i, scores[i])
            except:
                print("Something wrong!")
                # print(refs[i].shape)
                # print(refs[i].shape)
        return sum(scores)/len(scores)
    else:
        if rate == 16000:
            return pesq(rate, ref, deg, 'wb')
        # elif rate == 8000:
        #     return pesq(rate, ref, deg, 'nb')
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
    trimmed_duration = arg.trimmed_duration
    if arg.down_sample == 1:
        down_sample = True
    else:
        down_sample = False

    clean, fs = load_sample(path=arg.clean, down_sample=down_sample)
    denoised, fs = load_sample(arg.denoised, down_sample=down_sample)

    min_length = min(clean.shape[0], denoised.shape[0])
    clean = clean[:min_length]
    denoised = denoised[:min_length]

    # Print the result to console
    print("Clean voice: ", arg.clean)
    print("Denoised voice: ", arg.denoised)
    if arg.metric == 'pesq':
        print("Calculating PESQ score...")
        print("Done! PESQ score: ", eval_pesq(fs, clean, denoised, trimmed_duration))
    elif arg.metric == 'stoi':
        print("Evaluation with STOI metric is being maintained, please use PESQ instead")
        # print("|  STOI score: ", eval_stoi(fs, clean, denoised))
    else:
        print("|  The given metric isn't supported!")

if __name__ == '__main__':
    run_eval()