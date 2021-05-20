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
from tqdm import tqdm

from dataset import ASVspoof2019
from train import shuffle


def read_sfile(mname):
    data_np = np.genfromtxt(os.path.join(args.saved_path, mname), dtype=str)
    cols = ['fname', 'score']
    df = pd.DataFrame(index=data_np[:, 0], data=data_np, columns=cols)
    df['score'] = df['score'].astype(np.float32, copy=False)

    return df


def avg_fuse(args):
    frames = [read_sfile(m) for m in args.input]
    merge_cols = ['fname']
    fuse_result = pd.concat(frames).groupby(merge_cols, as_index=False)['score'].mean()

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
    score_list = [read_sfile(f) for f in args.input]
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
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--saved_path', type=str, default='/data/xinhui/scores/')

    args = parser.parse_args()
    fuse_result = avg_fuse(args)
    fuse_result = weighted_fuse(args)
    #   fuse_result.to_csv(args.output, sep=' ', header=False, index=False)
    print(fuse_result)