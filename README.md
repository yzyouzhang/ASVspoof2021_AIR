# ASVspoof2021_AIR
AIR system for ASVspoof 2021

This repository contains our implementation of the paper, "One-class Learning towards Generalized Voice Spoofing Detection".
[Paper link here](https://arxiv.org/pdf/2010.13995.pdf)
## Requirements
python==3.6

pytorch==1.1.0

## Data Preprocess
The raw wave files of the ASVspoof 2019 dataset is stored in `/data/neil/DS_10283_3336`.
The `raw_dataset.py` is a pytorch dataloader for the original speech dataset.
The `feature_extraction.py` is to write feature extraction methods, currently I only have LFCC calculation. You may include CQCC, melspectrogram, CQT, MFCC, etc. in the future for model fusion. The `utils_dsp.py` are some helper functions for feature extraction.
The `process.py` is using the functions in feature extraction and output the preprocessed speech dataset.

## Data Preparation
The `dataset.py` is a pytorch dataloader for the preprocessed speech dataset. 
Make sure you change the directory path to the path on your machine.

## Model and loss functions
The `model.py` include some model architectures.
The `loss.py` defines some loss functions, including what we proposed.

## Run the training code
In the `train.py`, please read the argparse to get familiar with the options.
Before running the `train.py`, please change the `path_to_features` according to the files' location on your machine.
```
python3 train.py --add_loss ang_iso -o /data/xinhui/models/ocsoftmax --gpu 3
```
## Run the test code with trained model
I included the test code in `utils.py`, you can write a `test.py` with argparse, similar to my public repository. The `eval_metrics.py` and `evaluate_tDCF_asvspoof19.py` are some helper functions to calculate the evaluation metrics.
Before running, please change the `model_dir` to the location of the model you would like to test with.
```
python3 utils.py
```
