# TonSpeech
Audio processing project 

## Evaluation on Noise suppression task
PESQ and STOI are currently supported metrics to evaluate the quality of denoising model. To evaluate test sample, run ```eval.sh```. <br />
Options:<br />
* ```--metric```: ```pesq``` or ```stoi```. For ```pesq```, please make sure that the sample rate of the given audio file is 8k (for narrow band) or 16k (for wide band, wide band is default option). 
* ```--clean```: path to the clean audio file (target)
* ```--denoised```: path to the audio after being denoised (result of model).


