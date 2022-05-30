# TonSpeech
Audio processing project 

## Installation
**Prerequisite** <br />
* python 3.6 or higher
* Ubuntu 18 or higher
  
(Optional) Create your virtual enviroment via script <br />
```python3 -m venv [your_venv_name]```<br />
then activate it <br />
```source [your_venv_name]/bin/activate``` </br>

**Install requirements** <br />
In your terminal, run <br />
``` git clone https://github.com/tungedng2710/TonSpeech.git```<br />
``` cd TonSpeech ``` <br />
``` pip install -r requirements.txt```


## Evaluation on Noise suppression task
PESQ and STOI are currently supported metrics to evaluate the quality of denoising model. To evaluate test sample, run ```eval.sh```. <br />
Options:<br />
* ```--metric```: ```pesq``` or ```stoi```. For ```pesq```, please make sure that the sample rate of the given audio file is 8k (for narrow band) or 16k (for wide band, wide band is default option). 
* ```--clean```: path to the clean audio file (target)
* ```--denoised```: path to the audio after being denoised (result of model).
