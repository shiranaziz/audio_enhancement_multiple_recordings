# Audio Enhancement from Multiple Crowdsourced Recordings
TThis code is released alongside the paper "Audio Enhancement from Multiple Crowdsourced Recordings: A Simple and Effective Baseline".
The code includes both the dataset creation process and the method described in the paper.
You can listen to audio samples on the online demo here: https://shiranaziz.github.io/examples/

## Installation
The code requires Python 3.8 (best install with conda) you can run the following comands:
```shell
pip install -r requirements.txt 
```

# Dataset Creation 
use the config_create_dataset.yaml for the input aguments:
```python

# choose betweeen predifined options 
# SOURCE_MUSIC_NOISE_SPEECH / SOURCE_MUSIC_NOISE_AUDIOSET / SOURCE_MUSIC_NOISE_DEMAND / 
# SOURCE_SPEECH_NOISE_SPEECH / SOURCE_SPEECH_NOISE_AUDIOSET / SOURCE_SPEECH_NOISE_DEMAND / 
# SOURCE_MUSIC_NOISE_SPEECH_CHANGED_NUMBER_OF_NOISES
dataset_type: SOURCE_MUSIC_NOISE_SPEECH
# The root directory to the data set for the source signal, the enhenced signal. choose betweeen music from MUSDB18 or speech from LibriSpeech
source_dataset_path: "path/to/musdb18/"
# The root directory to the data set for the noises.  choose betweeen speech from LibriSpeech or noises from audioset or DEMAND
noise_dataset_path:  "path/to/LibriSpeech/"
# The loction to store the dataset
output_dataset_path: "path/to/output_data/"
# create the data set with random one sec peacketloss at each one of the noises. boolean argumnt.
is_packetloss: False 
```
Then you can run the code in the comnd line:
```shell
python --config_path=config_create_dataset.yaml
```
 The output will be the path to the dataset. 
 
# Run The Experiment
```python
# choose betweeen predifined options
# SOURCE_MUSIC_NOISE_SPEECH / SOURCE_MUSIC_NOISE_AUDIOSET / SOURCE_MUSIC_NOISE_DEMAND /
# SOURCE_SPEECH_NOISE_SPEECH / SOURCE_SPEECH_NOISE_AUDIOSET / SOURCE_SPEECH_NOISE_DEMAND /
# SOURCE_SPEECH_NOISE_SPEECH_PACKETLOSS / SOURCE_SPEECH_NOISE_DEMAND_PACKETLOSS / SOURCE_MUSIC_NOISE_SPEECH_PACKETLOSS /
# SOURCE_MUSIC_NOISE_SPEECH_CHANGED_NUMBER_OF_NOISES
dataset_type: SOURCE_SPEECH_NOISE_SPEECH
# the output from the dataset creation stage
dataset_path: "path/to/output_data/snr_dataset/"
# the path for our methd and baselines signals, metreces csv and plots
output_dataset_path: "path/to/results/"

method_to_examine: [MEAN, MEDIAN, MAX_ELIMINATION, OURS]
metrics_to_examine: [SI_SNR, PESQ, STOI]

```


Then you can run the code in the comnd line:
```shell
python --config_path=config_experiment.yaml
```
