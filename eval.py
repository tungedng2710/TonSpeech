from scipy.io import wavfile
from pesq import pesq   
import soundfile as sf
from pystoi import stoi
import argparse 
import os

def get_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clean', type=str, help='path/to/clean/voice')
    parser.add_argument('--denoised', type=str, help='path/to/denoised/voice')
    parser.add_argument('--metric', type=str, help='pesq or stoi', default='pesq')
    return parser.parse_args()

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
    arg = get_arg()
    fs, clean = wavfile.read(arg.clean)
    fs, denoised = wavfile.read(arg.denoised)
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