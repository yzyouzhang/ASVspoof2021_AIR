import argparse
import os
import shutil
import math

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from numpy import array
import eval_metrics as em


def read_file(fname):
    data_np = np.genfromtxt(fname, dtype=str)
    cols = ['fname', 'sysid', 'key', 'score']
    df = pd.DataFrame(index=data_np[:,0],data=data_np,columns=cols)
    df['score']=df['score'].astype(np.float32, copy=False)
    return df


def avg_fuse(args):
    frames = [read_file(f) for f in args.input]
    merge_cols = ['fname', 'sysid', 'key']
    result_df = pd.concat(frames).groupby(merge_cols, as_index=False)['score'].sum()
    result_df.to_csv(args.output + 'avg_fuse_score', sep=' ', header=False, index=False)
    print('done')

    return result_df


def weighted_fuse(args):
    weight = []
    frames = [read_file(f) for f in args.input]
    merge_cols = ['fname', 'sysid', 'key']
    weight = cal_weight(args)
    
    for i in range(len(frames)):
        frames[i]['score'] = frames[i]['score']*weight[i]
    result_df = pd.concat(frames).groupby(merge_cols, as_index=False)['score'].mean()
    result_df.to_csv(args.output + 'avg_fuse_score', sep=' ', header=False, index=False)
    print('done')
    
    return result_df
    
def cal_weight(args):
    
    weight = []
    
    for i in range(len(args.input)):
        with open('/data/xinhui/scores/model_eers') as f:
            for line in f:
                feat = line.split()[0]
                eer = line.split()[1]
                
                if feat in args.input[i]:
                    weight.append(eer)
    
    weight = list(map(float,weight))
    print("original eers:", weight)
    max_w = max(weight)
    min_w = min(weight)
    
    if max_w == min_w:
        print("equal weight")
        pass
    else:
        for i in range(len(weight)):
            weight[i] = (max_w-weight[i])/(max_w-min_w)
            if weight[i]==0:
                weight[i]=0.00001
            else:
                pass
  
        k = 1.0/math.log(len(weight))
        lnf = [None for i in range(len(weight))]
        lnf = array(lnf)
        for i in range(len(weight)):
            if weight[i]==0:
                lnfi = 0.0
            else:
                p = weight[i]/sum(weight)
                lnfi = math.log(p) * p * (-k)
            weight[i] = 1-lnfi
        sum_w = sum(weight)   
    
        for i in range(len(weight)):
            weight[i] = weight[i]/sum_w
    
    return weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Result Fusion Utility')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='models for fusion')
    parser.add_argument('-o', '--output', type=str, help="output folder", default='/data/xinhui/scores/fuse_scores/')
    parser.add_argument('-m', '--method', type=str, help='fusion method', required=True, choices=['avg', 'wght'])
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
            os.makedirs(args.output)
    print('Processing input files :', args.input)

    if args.method=='avg':
        fuse_result = avg_fuse(args)
        
    elif args.method=='wght':
        fuse_result = weighted_fuse(args)
    
    #   fuse_result.to_csv(args.output, sep=' ', header=False, index=False)

    target_scores, nontarget_scores = [], []
    fuse_result0 = fuse_result[fuse_result['key'] == 'bonafide']
    fuse_result1 = fuse_result[fuse_result['key'] == 'spoof']
    target_scores = (fuse_result0['score'])
    nontarget_scores = (fuse_result1['score'])

    eer = em.compute_eer(target_scores, nontarget_scores)[0]
    other_eer = em.compute_eer(-target_scores, -nontarget_scores)[0]
    eer = min(eer, other_eer)
    print(eer)
