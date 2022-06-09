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
* ```--clean```: path to the clean audio file (target)
* ```--denoised```: path to the audio after being denoised (result of model).

## Speech Enhancement with MetricGAN+
Official implementation of MetricGAN+ at this [GitHub link](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN)

In the terminal, run the script below
```bat
python se_metricganplus.py --noisy [path/to/noisy/audio/or/folder]
```
The given path will be automatically check whether it is a file or folder.
