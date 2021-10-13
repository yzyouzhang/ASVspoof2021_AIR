# ASVspoof2021_AIR
## UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021


This repository contains our implementation of the paper, "UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021".
[Paper link here](https://www.isca-speech.org/archive/asvspoof_2021/chen21_asvspoof.html)
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

## Citations
```
@inproceedings{chen21_asvspoof,
  author={Xinhui Chen and You Zhang and Ge Zhu and Zhiyao Duan},
  title={{UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021}},
  year=2021,
  booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
  pages={75--82},
  doi={10.21437/ASVSPOOF.2021-12}
}
```

## Reference to our prior works
[1] Y. Zhang, F. Jiang and Z. Duan, "One-Class Learning Towards Synthetic Voice Spoofing Detection," in IEEE Signal Processing Letters, vol. 28, pp. 937-941, 2021, doi: 10.1109/LSP.2021.3076358. [[link](https://ieeexplore.ieee.org/document/9417604)] [[code](https://github.com/yzyouzhang/AIR-ASVspoof)]

[2] Y. Zhang, G. Zhu, F. Jiang, Z. Duan, An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems. Proc. Interspeech 2021, pp. 4309-4313, 2021, doi: 10.21437/Interspeech.2021-1820. [[link](https://www.isca-speech.org/archive/interspeech_2021/zhang21ea_interspeech.html)] [[code](https://github.com/yzyouzhang/Empirical-Channel-CM)]

