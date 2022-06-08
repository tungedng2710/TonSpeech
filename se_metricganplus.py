# Speech Enhancement with MetricGAN+
# Official repository: https://github.com/speechbrain/speechbrain

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
import timeit
import os
from tqdm import tqdm

NOISE_PATH = "./exp/noise30s"

def enhance(enhance_model, noise_path):
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
    if not os.path.exists("metricgan_exp"):
        os.makedirs("metricgan_exp")
    start = timeit.default_timer()
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )
    enhance(enhance_model=enhance_model, 
            noise_path=NOISE_PATH)
    stop = timeit.default_timer()
    print('Time: ', stop - start)


