# TonSpeech
**Audio processing project**

## Installation
**Prerequisite** <br />
* python 3.7 or higher
* Ubuntu 18 or higher
  
(Optional) Create your virtual enviroment via script <br />
```bat 
python3 -m venv [your_venv_name]
```
then activate it <br />
```bat
source [your_venv_name]/bin/activate
``` 

**Install requirements** <br />
In your terminal, run <br />
```bat 
git clone https://github.com/tungedng2710/TonSpeech.git
cd TonSpeech 
pip install -r requirements.txt
```


## Evaluation on Noise suppression task
PESQ and STOI are currently supported metrics to evaluate the quality of denoising model. To evaluate test sample, run 
```bat
bash run_eval.sh
```
Options:<br />
* ```--trimmed_duration```: (optional for long audio) length of sample batch (seconds), default value is ```-1``` (no trimming) 
* ```--down_sample```: 1 (True) or 0 (False)
* ```--metric```: ```pesq``` or ```stoi```. For ```pesq```, please make sure that the sample rate of the given audio file is 8k (for narrow band) or 16k (for wide band, wide band is default option). 
* ```--clean```: path to the clean audio file (target). If you need to evaluate on Voicebank-DEMAND dataset, it will be the path to clean audio folder.
* ```--denoised```: path to the audio after being denoised (result of model).
* ```--eval_on_dataset```: ```0```: run with single audio (denoised-clean); ```1```: compare a clean audio with a folder of different denoised audio; ```2```: eval on Voicebank-DEMAND.

## Speech Enhancement with MetricGAN+
Official implementation of MetricGAN+ at this [GitHub link](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN)

In the terminal, run the script below
```bat
python se_metricganplus.py --noisy [path/to/noisy/audio/or/folder]
```
The given path will be automatically check whether it is a file or folder. <br />
If you need to post-process the output of MetricGAN+ with Perceptual Contrast Stretching ([PCS](https://github.com/RoyChao19477/PCS)), run
```bat
python se_pcs.py --noisy [folder/of/metricganplus_results]
```
## ONNX model
**TonSpeech** supports exporting MetricGAN+ to ONNX model by modifying short-time Fourier transform operator (Unfortunately, torch.stft and torch.istft are not supported in current opset version). To export onnx model, just run
```bash
python onnx.py
```
Dummy input is located in ```data``` folder. Currently, you shouldn't use another dummy input because it is related to the fixed signal length of Fourier transform operator. It will be fixed soon.
## Other ways to test audio quality
Currently, **TonSpeech** only supports to evaluate with **PESQ** and **STOI** metric, while other methods are developing. There are some alternative ways that you can try: <br />
* [ViSQOL](https://github.com/google/visqol) (Virtual Speech Quality Objective Listener): Being developed by Google. Similar to PESQ, ViSQOL evaluate the quality of audio by comparing between reference (clean) and degraded (denoised) audio and then map the result to MOS score.
* [CSIG,CBAK,COVL](https://github.com/usimarit/semetrics): Popular metrics of Speech Enhancement task on paperswithcode.
