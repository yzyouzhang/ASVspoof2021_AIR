import argparse
import os
import shutil
import torch
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader

from dataset import ASVspoof2019


def read_file(filename):
    fp = open(filename)
    score_list = []
    for line in fp.readlines():
        beg = line.find('[')
        end = line.find(']')
        line = line[beg + 1:end]
        data = line.split()
        score_list = score_list + data
    for i in range(3):
        score_list[i] = float(score_list[i])
    score_list = score_list[:3]
    score_list = pd.DataFrame(score_list)

    return score_list


def avg_fuse(file_list):
    score_list = [read_file(f) for f in file_list]
    df = pd.concat(score_list, axis=1)
    fuse_result = df.mean(1)

    return fuse_result


def weighted_fuse(arg):
    training_set = ASVspoof2019(args.access_type, args.path_to_features, 'train',
                                args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, args.path_to_features, 'dev',
                                  args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    trainDataLoader = DataLoader(training_set, batch_size=int(args.batch_size * args.ratio),
                                 shuffle=True, num_workers=args.num_workers, collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, collate_fn=validation_set.collate_fn)
    test_set = ASVspoof2019(args.access_type, args.path_to_features, "eval", args.feat, feat_len=args.feat_len,
                            pad_chop=args.pad_chop, padding=args.padding)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=test_set.collate_fn)
    models = []
    score_list = [read_file(f) for f in args.input]
    for m in range(len(args.imput)):
        models.append(torch.load(args.input[m]))
    eclf = VotingClassifier(estimators=[models], voting='soft')
    eclf = eclf.fit(training_set, validation_set)
    weights = eclf.get_params
    weights = pd.DataFrame(weights)
    fuse_result = score_list.mul(weights).mean(1)

    return fuse_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Result Fusion Utility')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='models for fusion')
    parser.add_argument('-o', '--output', type=str, help="output folder")
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/neil/ASVspoof2019LA/')
    parser.add_argument("--feat", type=str, help="which feature to use", default='LFCC',
                        choices=["CQCC", "LFCC", "MFCC", "STFT", "Melspec", "CQT", "LFB", "LFBB"])
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=True,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument("--ratio", type=float, default=1,
                        help="ASVspoof ratio in a training batch, the other should be external genuine speech")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    args = parser.parse_args()
    fuse_result = avg_fuse(args.input)
    fuse_result = weighted_fuse(args)
#   fuse_result.to_csv(args.output, sep=' ', header=False, index=False)
    print(fuse_result)
