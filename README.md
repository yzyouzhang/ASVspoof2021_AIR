# ASVspoof2021_AIR
## 


This repository contains our implementation of the paper, "".
[Paper link here]()
### Requirements
python==3.6

pytorch==1.1.0

### Data Preprocess
The raw wave files of the ASVspoof 2019 dataset is stored in `/data/neil/DS_10283_3336`.
The `raw_dataset.py` is a pytorch dataloader for the original speech dataset.
The `feature_extraction.py` is to write feature extraction methods, currently I only have LFCC calculation. You may include CQCC, melspectrogram, CQT, MFCC, etc. in the future for model fusion. The `utils_dsp.py` are some helper functions for feature extraction.
The `process.py` is using the functions in feature extraction and output the preprocessed speech dataset.

### Data Preparation
The `dataset.py` is a pytorch dataloader for the preprocessed speech dataset. 
Make sure you change the directory path to the path on your machine.

### Model and loss functions
The `model.py` include some model architectures.
The `loss.py` defines some loss functions, including what we proposed.

### Run the training code
In the `main_train.py`, please read the argparse to get familiar with the options.
Before running the `main_train.py`, please change the `path_to_features` according to the files' location on your machine.
```
python3 main_train.py --add_loss ang_iso -o /data/xinhui/models/ocsoftmax --gpu 3
```
### Run the test code with trained model
I included the test code in `generate_score.py`. The `eval_metrics.py` and `evaluate_tDCF_asvspoof19.py` are some helper functions to calculate the evaluation metrics.
Before running, please change the `model_dir` to the location of the model you would like to test with.
```
python3 generate_score.py
```

## 

