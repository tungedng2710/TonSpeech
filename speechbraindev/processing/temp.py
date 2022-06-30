import torch
import torchlibrosa as tl
import warnings
warnings.filterwarnings("ignore")

real = torch.load("real.pt")
imag = torch.load("imag.pt")

win_length = 256
hop_length = 512
n_fft = 512
istft_extractor = tl.ISTFT(n_fft=n_fft, 
                            hop_length=hop_length, 
                            win_length=win_length,
                            onnx=True)

sig_length = 27861
istft = istft_extractor(real, imag, length=sig_length)

torch.save(istft, "istft.pt")