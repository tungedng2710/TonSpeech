# TonSpeech
Audio processing project 

## Installation
**Prerequisite** <br />
* python 3.6 or higher
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
PESQ and STOI are currently supported metrics to evaluate the quality of denoising model. To evaluate test sample, run ```eval.sh```. <br />
Options:<br />
* ```--metric```: ```pesq``` or ```stoi```. For ```pesq```, please make sure that the sample rate of the given audio file is 8k (for narrow band) or 16k (for wide band, wide band is default option). 
* ```--clean```: path to the clean audio file (target)
* ```--denoised```: path to the audio after being denoised (result of model).
