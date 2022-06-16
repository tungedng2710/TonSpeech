from pesq import pesq   
from pystoi import stoi
import argparse 
from tqdm import tqdm
import os

import torch
import torchaudio
from torchaudio.transforms import Resample

def load_sample(path: str = None,
                down_sample: bool = False):
    """
    Stereo audio will be converted to mono audio automatically
    path (str): path/to/your/audio/file
    down_sample (bool): change sample rate into 16 kHz
    """
    assert path is not None
    signal, rate = torchaudio.load(path)
    if len(signal) == 2:
        # print("Warning: {fname} is stereo audio".format(fname = os.path.basename(path)))
        signal = signal[0]

    if down_sample:
        new_rate = 16000
        downsampler = Resample(orig_freq=rate, new_freq=new_rate)
        downsampled_signal = downsampler(signal)
        signal = downsampled_signal
        rate = new_rate
    else:
        print("Warning: sample rate = ", rate)

    signal = torch.flatten(signal).numpy()
    return signal, rate

def get_batches(data, rate, duration):
    '''
    split audio data into batches
    data: (numpy.ndarray): signal vector of audio file
    rate: (int): sample frequency (example: 16000 is equivalent to 16kHz)
    duration: (int): duration of batch audio (seconds)
    '''
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
    clean: clean voice
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