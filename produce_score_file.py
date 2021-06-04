import argparse
import os
import collections
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ASVspoof2019
from utils import *
from loss import *

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--pretrain_out_fold', type=str, help="directory for pretrained model", default='/data/xinhui/models/')
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path", default='/data2/neil/ASVspoof2019LA/')
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=True, help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'], help="how to pad short utterance")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for isolate loss")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for isolate loss")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for angular isolate loss")
    parser.add_argument('--save_path', type=str, default='/data/xinhui/scores/')
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def produce_lfcc_file(args):
    model = torch.load(os.path.join(args.pretrain_out_fold,'lfcc','checkpoint','anti-spoofing_cqcc_model_64.pt'))
    test_set = ASVspoof2019(args.access_type, args.path_to_features, "eval", 'LFCC', feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=test_set.collate_fn)

    model.eval()
    with torch.no_grad():
        tag_loader, idx_loader, label_loader, fname_loader, score_loader = [], [], [], [], []

        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.transpose(2, 3).to(args.device)
            tags = tags.to(args.device)
            labels = labels.to(args.device)

            feats, outs = model(cqcc)
            score = F.softmax(outs, dim=1)[:, 0]

            label_loader.append(labels)
            idx_loader.extend(['spoof' if l==1 else 'bonafide' for l in labels])
            tag_loader.append(tags)
            fname_loader.extend(list(audio_fn))
            score_loader.append(score)
        
        labels = torch.cat(label_loader, 0).data.cpu().numpy()
        tags = torch.cat(tag_loader, 0).data.cpu().numpy()
        scores = torch.cat(score_loader, 0).data.cpu().numpy()

        eer = em.compute_eer(scores[labels==0], scores[labels==1])[0]
        other_eer = em.compute_eer(-scores[labels==0], -scores[labels==1])[0]
        eer = min(eer, other_eer)
        print(eer)

        with open(os.path.join(args.save_path, 'lfcc_score'),'w') as f:
            for fn, tg, key, cm in zip(fname_loader, tags, idx_loader, scores):
                f.write('{} {} {} {}\n'.format(fn, tg, key, cm))

    print('done')

    return eer

def produce_stft_file(args):
    model = torch.load(os.path.join(args.pretrain_out_fold,'stft','checkpoint','anti-spoofing_cqcc_model_64.pt'))
    test_set = ASVspoof2019(args.access_type, args.path_to_features, "eval", 'STFT', feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=test_set.collate_fn)
    model.eval()
    with torch.no_grad():
        label_loader, tag_loader, idx_loader, fname_loader, score_loader = [], [], [], [], []

        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.transpose(2, 3).to(args.device)
            tags = tags.to(args.device)
            labels = labels.to(args.device)
            feats, outs = model(cqcc)
            score = F.softmax(outs, dim=1)[:, 0]

            idx_loader.extend(['spoof' if l==1 else 'bonafide' for l in labels])
            tag_loader.append(tags)
            fname_loader.extend(list(audio_fn))
            score_loader.append(score)
            label_loader.append(labels)
            
        labels = torch.cat(label_loader, 0).data.cpu().numpy()
        tags = torch.cat(tag_loader, 0).data.cpu().numpy()
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        
        eer = em.compute_eer(scores[labels==0], scores[labels==1])[0]
        other_eer = em.compute_eer(-scores[labels==0], -scores[labels==1])[0]
        eer = min(eer, other_eer)
        print(eer)

        with open(os.path.join(args.save_path, 'stft_score'),'w') as f:
            for fn, tg, key, cm in zip(fname_loader, tags, idx_loader, scores):
                f.write('{} {} {} {}\n'.format(fn, tg, key, cm))

    print('done')
    return eer

if __name__ == "__main__":
    args = init()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    features, eers = [], []
    features.append('lfcc')
    eers.append(produce_lfcc_file(args))
    features.append('stft')
    eers.append(produce_stft_file(args))

    with open(os.path.join(args.save_path, 'model_eers'),'w') as f:
            for feat, eer in zip(features, eers):
                f.write('{} {}\n'.format(feat, eer))
