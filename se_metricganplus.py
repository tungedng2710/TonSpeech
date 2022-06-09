# Speech Enhancement with MetricGAN+
# Official repository: https://github.com/speechbrain/speechbrain

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
import timeit
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--noisy", type=str, help="path/to/noisy/voice/(or the folder of noisy voice)")
parser.add_argument("--saved_folder", type=str, help="path/to/output/folder", default="metricgan_exp")
args = parser.parse_args()

def enhance(enhance_model, noise_path):
    """
    noise_path should be path to audio file or folder of audio file
    """
    if os.path.isfile(noise_path):
        noisy = enhance_model.load_audio(noise_path).unsqueeze(0)
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
        fname = "denoised"+os.path.basename(noise_path)
        torchaudio.save("metricgan_exp/"+fname, enhanced.cpu(), 16000)
    else:
        for file in tqdm(os.listdir(noise_path)):
            fpath = noise_path+'/'+file
            noisy = enhance_model.load_audio(fpath).unsqueeze(0)
            enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
            fname = "denoised_"+os.path.basename(fpath)
            torchaudio.save("metricgan_exp/"+fname, enhanced.cpu(), 16000)
            os.remove(file)

if __name__ == "__main__":
    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder)
    start = timeit.default_timer()
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )
    enhance(enhance_model=enhance_model, 
            noise_path=args.noisy)
    stop = timeit.default_timer()
    print('Inference time: {t}s'.format(t=round(stop-start, 2)))


