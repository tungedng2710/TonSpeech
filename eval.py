# Copyright (c) 2022 TonAI Research, MetaTon
# Author: Tung Ng
# Our GitHub: https://github.com/MetaTon-AI-Research

import argparse 
import os
from eval_utils import load_sample, eval_pesq, eval_stoi
import pandas as pd
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--eval_on_dataset', type=int, help='0: single sample; 1: dataset', default=0)
    parser.add_argument('--down_sample', type=int, help='down sample rate into 16k Hz', default=1)
    parser.add_argument('--clean', type=str, help='path/to/clean/voice')
    parser.add_argument('--denoised', type=str, help='path/to/denoised/voice (single) or folder (dataset)')
    parser.add_argument('--metric', type=str, help='pesq or stoi', default='pesq')
    parser.add_argument('--trimmed_duration', type=int, default=-1)
    parser.add_argument('--to_csv', type=int, help='save the results to csv file', default=0)
    parser.add_argument('--verbose', type=int, help='show the progress', default=1)
    
    return parser.parse_args()

def eval_single_sample(args):
    trimmed_duration = args.trimmed_duration
    if args.down_sample == 1:
        down_sample = True
    else:
        down_sample = False

    clean, fs = load_sample(path=args.clean, down_sample=down_sample)
    denoised, fs = load_sample(path=args.denoised, down_sample=down_sample)

    min_length = min(len(clean), len(denoised))
    clean = clean[:min_length]
    denoised = denoised[:min_length]

    # Print the result to console
    print("Clean voice: ", args.clean)
    print("Denoised voice: ", args.denoised)
    if args.metric == 'pesq':
        print("Calculating PESQ score...")
        print("PESQ score: ", eval_pesq(fs, clean, denoised, trimmed_duration))
    elif args.metric == 'stoi':
        print("STOI score: ", eval_stoi(fs, clean, denoised))
    else:
        print("Metric should be pesq or stoi")

def eval_dataset(args):
    trimmed_duration = args.trimmed_duration
    if args.down_sample == 1:
        down_sample = True
    else:
        down_sample = False

    root_dir = args.denoised
    sample_files = sorted(os.listdir(root_dir))
    clean, fs = load_sample(path=args.clean, down_sample=down_sample)
    scores = []
    fnames = []
    for file in tqdm(sample_files):
        fnames.append(file)
        fpath = root_dir+'/'+file
        denoised, fs = load_sample(path=fpath, down_sample=down_sample)
        min_length = min(len(clean), len(denoised))
        clean = clean[:min_length]
        denoised = denoised[:min_length]
        if args.metric == 'pesq':
            score = eval_pesq(fs, clean, denoised, trimmed_duration)
            scores.append(score)
        elif args.metric == 'stoi':
            score = eval_stoi(fs, clean, denoised)
            scores.append(score)
        else:
            print("Metric should be pesq or stoi")
            break

    if args.verbose > 0:
        print("---------------------------------------------")
        for i in range(len(fnames)):
            print("{file}: PESQ: {score}".format(file=fnames[i], score=round(scores[i], 4)))
        print("Mean PESQ: {score}".format(score=round(sum(scores)/len(scores), 4)))

    if args.to_csv == 1:
        print("Saving result to csv file...")
        df_dict = {"file_name": fnames,
                   args.metric+"_score": scores}
        df = pd.DataFrame(df_dict)
        saved_name = args.metric+"_results.csv"
        df.to_csv(saved_name, index=False)

if __name__ == "__main__":
    args = get_args()
    if args.eval_on_dataset == 0:
        eval_single_sample(args)
    else:
        print("Evaluating...")
        eval_dataset(args)
        print("Done!")