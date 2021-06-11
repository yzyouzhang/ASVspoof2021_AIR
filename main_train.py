import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from loss import *
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path", default='/data/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/neil/ASVspoof2019LA/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')
    parser.add_argument("-e", "--path_to_external", type=str, help="external data for training",
                        default="/dataNVME/neil/libriTTS/train-clean-360")

    parser.add_argument("--ratio", type=float, default=0.5,
                        help="ASVspoof ratio in a training batch, the other should be augmented")

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='LFCC',
                        choices=["CQCC", "LFCC", "MFCC", "STFT", "Melspec", "CQT", "LFB", "LFBB"])
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=True, help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    parser.add_argument('-m', '--model', help='Model arch', default='lcnn',
                        choices=['cnn', 'resnet', 'lcnn', 'tdnn', 'lstm', 'rnn', 'cnn_lstm'])

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.7, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=50, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"], help="use which loss for basic training")
    parser.add_argument('--add_loss', type=str, default=None,
                        choices=[None, 'center', 'lgm', 'lgcl', 'isolate', 'iso_sq', 'ang_iso', 'multi_isolate', 'multicenter_isolate'], help="add other loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for isolate loss")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for isolate loss")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for angular isolate loss")
    parser.add_argument('--num_centers', type=int, default=3, help="num of centers for multi isolate loss")

    parser.add_argument('--visualize', action='store_true', help="feature visualization")
    parser.add_argument('--test_only', action='store_true', help="test the trained model in case the test crash sometimes or another test method")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    parser.add_argument('--device_adv', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use device_adversarial in training")
    parser.add_argument('--LA_aug', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use LA_augmentation in training")
    parser.add_argument('--lambda_', type=float, default=0.1, help="lambda for gradient reversal layer")
    parser.add_argument('--lr_d', type=float, default=0.0001, help="learning rate")

    parser.add_argument('--pre_train', action='store_true', help="whether to pretrain the model")
    parser.add_argument('--add_genuine', action='store_true', help="whether to iterate through genuine part multiple times")
    parser.add_argument('--test_on_eval', action='store_true',
                        help="whether to run EER on the evaluation set")

    args = parser.parse_args()

    # Check ratio
    assert (args.ratio > 0) and (args.ratio <= 1)

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.test_only or args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    print(args.pad_chop)

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def shuffle(cqcc, tags, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    cqcc = cqcc[shuffle_index]
    tags = tags[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return cqcc, tags, labels

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'resnet':
        node_dict = {"CQCC": 4, "LFCC": 3, "LFBB": 3, "Melspec": 6, "LFB": 6, "CQT": 8, "STFT": 11, "MFCC": 87}
        cqcc_model = ResNet(node_dict[args.feat], args.enc_dim, resnet_type='18', nclasses=1 if args.base_loss == "bce" else 2).to(args.device)
    elif args.model == 'cnn':
        cqcc_model = ConvNet(num_classes = 2, num_nodes = 47232, enc_dim = 256).to(args.device)
    elif args.model == 'lcnn':
        cqcc_model = LCNN(60, args.enc_dim, nclasses=2).to(args.device)

    if args.continue_training:
        cqcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt')).to(args.device)
    # cqcc_model = nn.DataParallel(cqcc_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    cqcc_optimizer = torch.optim.Adam(cqcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019(args.access_type, args.path_to_features, 'train',
                                args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, args.path_to_features, 'dev',
                                  args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    if args.LA_aug:
        training_set = ASVspoof2021LA_aug(part="train",
                                        feature=args.feat, feat_len=args.feat_len,
                                        pad_chop=args.pad_chop, padding=args.padding)
        validation_set = ASVspoof2021LA_aug(part="dev",
                                        feature=args.feat, feat_len=args.feat_len,
                                        pad_chop=args.pad_chop, padding=args.padding)
    if args.device_adv:
        training_set = ASVspoof2019LA_DeviceAdversarial(path_to_features="/data2/neil/ASVspoof2019LA/",
                                                        path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice",
                                                        part="train",
                                                        feature=args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        # grl = GradientReversal(args.lambda_)
        classifier = ChannelClassifier(args.enc_dim, len(training_set.devices)+1, args.lambda_).to(args.device)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_d,
                                                betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
        validation_set = ASVspoof2019LA_DeviceAdversarial(path_to_features="/data2/neil/ASVspoof2019LA/",
                                                          path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice",
                                                          part="dev",
                                                          feature=args.feat, feat_len=args.feat_len,
                                                          pad_chop=args.pad_chop, padding=args.padding)
    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size * args.ratio),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(25380)))
    trainAugDataLoader = DataLoader(training_set, batch_size=args.batch_size - int(args.batch_size * args.ratio),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(25380, len(training_set))))
    trainOri_flow = iter(trainOriDataLoader)
    trainAug_flow = iter(trainAugDataLoader)

    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size * args.ratio),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(24844)))
    valAugDataLoader = DataLoader(validation_set, batch_size=args.batch_size - int(args.batch_size * args.ratio),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(24844, len(validation_set))))
    valOri_flow = iter(valOriDataLoader)
    valAug_flow = iter(valAugDataLoader)

    if args.add_genuine:
        training_genuine = ASVspoof2019(args.access_type, args.path_to_features, 'train',
                                        args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding, genuine_only=True)
        trainGenDataLoader = DataLoader(training_genuine, batch_size=int(args.batch_size * args.ratio), shuffle=True,
                                        num_workers=args.num_workers, collate_fn=training_genuine.collate_fn)

    test_set = ASVspoof2019(args.access_type, args.path_to_features, "eval", args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_set.collate_fn)

    feat, _, _, _, _ = training_set[23]
    print("Feature shape", feat.shape)

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        assert False

    if args.add_loss == "isolate":
        iso_loss = IsolateLoss(2, args.enc_dim, r_real=args.r_real, r_fake=args.r_fake).to(args.device)
        if args.continue_training:
            iso_loss = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt')).to(args.device)
        iso_loss.train()
        iso_optimzer = torch.optim.SGD(iso_loss.parameters(), lr=args.lr)

    if args.add_loss == "iso_sq":
        iso_loss = IsolateSquareLoss(2, args.enc_dim, r_real=args.r_real, r_fake=args.r_fake).to(args.device)
        if args.continue_training:
            iso_loss = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt')).to(args.device)
        iso_loss.train()
        iso_optimzer = torch.optim.SGD(iso_loss.parameters(), lr=args.lr)

    if args.add_loss == "ang_iso":
        ang_iso = AngularIsoLoss(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        ang_iso.train()
        ang_iso_optimzer = torch.optim.SGD(ang_iso.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_loss = 1e8
    add_size = args.batch_size - int(args.batch_size * args.ratio)

    if args.add_loss is None:
        monitor_loss = 'base_loss'
    else:
        monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        genuine_feats, ip1_loader, tag_loader, idx_loader = [], [], [], []
        cqcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, cqcc_optimizer, epoch_num)
        if args.add_loss == "isolate":
            adjust_learning_rate(args, args.lr, iso_optimzer, epoch_num)
        if args.add_loss == "ang_iso":
            adjust_learning_rate(args, args.lr, ang_iso_optimzer, epoch_num)
        if args.device_adv:
            adjust_learning_rate(args, args.lr_d, classifier_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m, correct_c, total_c, correct_v, total_v = 0, 0, 0, 0, 0, 0

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                cqccOri, audio_fnOri, tagsOri, labelsOri, channelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                cqccOri, audio_fnOri, tagsOri, labelsOri, channelsOri = next(trainOri_flow)

            try:
                cqccAug, audio_fnAug, tagsAug, labelsAug, channelsAug = next(trainAug_flow)
            except StopIteration:
                trainAug_flow = iter(trainAugDataLoader)
                cqccAug, audio_fnAug, tagsAug, labelsAug, channelsAug = next(trainAug_flow)

            cqcc = torch.cat((cqccOri, cqccAug), 0)
            tags = torch.cat((tagsOri, tagsAug), 0)
            labels = torch.cat((labelsOri, labelsAug), 0)
            channels = channelsOri + channelsAug

            # count = 0
            # for channel in list(channels):
            #     if channel == "no_channel":
            #         count += 1
            # print(count / 64)

            # if i > 2: break
            cqcc = cqcc.transpose(2,3).to(args.device)

            tags = tags.to(args.device)
            labels = labels.to(args.device)
            # this_len = this_len.to(args.device)

            if args.ratio < 1:
                cqcc, tags, labels = shuffle(cqcc, tags, labels)

            feats, cqcc_outputs = cqcc_model(cqcc)

            if args.base_loss == "bce":
                cqcc_loss = criterion(cqcc_outputs, labels.unsqueeze(1).float())
            else:
                cqcc_loss = criterion(cqcc_outputs, labels)

            trainlossDict['base_loss'].append(cqcc_loss.item())

            if args.add_loss == None:
                cqcc_optimizer.zero_grad()
                cqcc_loss.backward()
                cqcc_optimizer.step()

            if args.add_loss in ["isolate", "iso_sq"]:
                isoloss = iso_loss(feats, labels)
                cqcc_loss = isoloss * args.weight_loss
                cqcc_optimizer.zero_grad()
                iso_optimzer.zero_grad()
                trainlossDict[args.add_loss].append(isoloss.item())
                cqcc_loss.backward()
                cqcc_optimizer.step()
                iso_optimzer.step()

            if args.add_loss == "ang_iso":
                ang_isoloss, _ = ang_iso(feats, labels)
                cqcc_loss = ang_isoloss * args.weight_loss
                if epoch_num > 0 and args.device_adv:
                    channel = channel.to(args.device)
                    # feats = grl(feats)
                    classifier_out = classifier(feats)
                    _, predicted = torch.max(classifier_out.data, 1)
                    total_m += channel.size(0)
                    correct_m += (predicted == channel).sum().item()
                    device_loss = criterion(classifier_out, channel)
                    # print(cqcc_loss.item())
                    cqcc_loss += device_loss
                    # print(device_loss.item())
                    trainlossDict["adv_loss"].append(device_loss.item())
                cqcc_optimizer.zero_grad()
                ang_iso_optimzer.zero_grad()
                trainlossDict[args.add_loss].append(ang_isoloss.item())
                cqcc_loss.backward()
                cqcc_optimizer.step()
                ang_iso_optimzer.step()

            if args.device_adv:
                channel = channel.to(args.device)
                feats, _ = cqcc_model(cqcc)
                feats = feats.detach()
                # feats = grl(feats)
                classifier_out = classifier(feats)
                _, predicted = torch.max(classifier_out.data, 1)
                total_c += channel.size(0)
                correct_c += (predicted == channel).sum().item()
                device_loss_c = criterion(classifier_out, channel)
                classifier_optimizer.zero_grad()
                device_loss_c.backward()
                classifier_optimizer.step()


            # genuine_feats.append(feats[labels==0])
            ip1_loader.append(feats)
            idx_loader.append((labels))
            tag_loader.append((tags))

            # if epoch_num > 0:
            #     print(100 * correct_m / total_m)
            #     print(100 * correct_c / total_c)

            # desc_str = ''
            # for key in sorted(trainlossDict.keys()):
            #     desc_str += key + ':%.5f' % (np.nanmean(trainlossDict[key])) + ', '
            # t.set_description(desc_str)
            # print(desc_str)

            if epoch_num > 0 and args.device_adv:
                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                              str(trainlossDict["adv_loss"][-1]) + "\t" +
                              str(100 * correct_m / total_m) + "\t" +
                              str(100 * correct_c / total_c) + "\t" +
                              str(trainlossDict[monitor_loss][-1]) + "\n")
            else:
                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                              str(trainlossDict[monitor_loss][-1]) + "\n")

        if args.visualize and ((epoch_num+1) % 3 == 1):
            feat = torch.cat(ip1_loader, 0)
            labels = torch.cat(idx_loader, 0)
            tags = torch.cat(tag_loader, 0)
            if args.add_loss in ["isolate", "iso_sq"]:
                centers = iso_loss.center
            elif args.add_loss == "ang_iso":
                centers = ang_iso.center
            else:
                centers = torch.mean(feat[labels == 0], dim=0, keepdim=True)
            visualize(args, feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(), centers.data.cpu().numpy(),
                      epoch_num + 1, "Train")
        # print(len(what))
        # print(len(list(set(what))))
        # assert len(what) == len(list(set(what)))
        # Val the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        cqcc_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            # with trange(2) as v:
            # with trange(len(valDataLoader)) as v:
            #     for i in v:
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    cqccOri, audio_fnOri, tagsOri, labelsOri, channelsOri = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    cqccOri, audio_fnOri, tagsOri, labelsOri, channelsOri = next(valOri_flow)

                try:
                    cqccAug, audio_fnAug, tagsAug, labelsAug, channelsAug = next(valAug_flow)
                except StopIteration:
                    valAug_flow = iter(valAugDataLoader)
                    cqccAug, audio_fnAug, tagsAug, labelsAug, channelsAug = next(valAug_flow)

                cqcc = torch.cat((cqccOri, cqccAug), 0)
                tags = torch.cat((tagsOri, tagsAug), 0)
                labels = torch.cat((labelsOri, labelsAug), 0)
                channels = channelsOri + channelsAug

                # if i > 2: break
                cqcc = cqcc.transpose(2,3).to(args.device)

                tags = tags.to(args.device)
                labels = labels.to(args.device)

                cqcc, tags, labels = shuffle(cqcc, tags, labels)

                feats, cqcc_outputs = cqcc_model(cqcc)

                if args.base_loss == "bce":
                    cqcc_loss = criterion(cqcc_outputs, labels.unsqueeze(1).float())
                    score = cqcc_outputs[:, 0]
                else:
                    cqcc_loss = criterion(cqcc_outputs, labels)
                    score = F.softmax(cqcc_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))
                tag_loader.append((tags))

                if args.add_loss in [None]:
                    devlossDict["base_loss"].append(cqcc_loss.item())
                elif args.add_loss in ["lgm", "center"]:
                    devlossDict[args.add_loss].append(cqcc_loss.item())
                elif args.add_loss == "lgcl":
                    outputs, moutputs = lgcl_loss(feats, labels)
                    cqcc_loss = criterion(moutputs, labels)
                    score = F.softmax(outputs, dim=1)[:, 0]
                    devlossDict[args.add_loss].append(cqcc_loss.item())
                elif args.add_loss in ["isolate", "iso_sq"]:
                    isoloss = iso_loss(feats, labels)
                    score = torch.norm(feats - iso_loss.center, p=2, dim=1)
                    devlossDict[args.add_loss].append(isoloss.item())
                elif args.add_loss == "ang_iso":
                    ang_isoloss, score = ang_iso(feats, labels)
                    devlossDict[args.add_loss].append(ang_isoloss.item())
                    if epoch_num > 0 and args.device_adv:
                        channel = channel.to(args.device)
                        # feats = grl(feats)
                        classifier_out = classifier(feats)
                        _, predicted = torch.max(classifier_out.data, 1)
                        total_v += channel.size(0)
                        correct_v += (predicted == channel).sum().item()
                        device_loss = criterion(classifier_out, channel)
                        devlossDict["adv_loss"].append(device_loss.item())

                score_loader.append(score)

                # desc_str = ''
                # for key in sorted(devlossDict.keys()):
                #     desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                # # v.set_description(desc_str)
                # print(desc_str)
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            eer = min(eer, other_eer)

            if epoch_num > 0 and args.device_adv:
                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t"+ "\t" +
                              str(np.nanmean(devlossDict["adv_loss"])) + "\t" +
                              str(100 * correct_v / total_v) + "\t" +
                              str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                              str(eer) + "\n")
            else:
                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" +
                              str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                              str(eer) +"\n")
            print("Val EER: {}".format(eer))

            if args.visualize and ((epoch_num+1) % 3 == 1):
                feat = torch.cat(ip1_loader, 0)
                tags = torch.cat(tag_loader, 0)
                if args.add_loss == "isolate":
                    centers = iso_loss.center
                elif args.add_loss == "ang_iso":
                    centers = ang_iso.center
                else:
                    centers = torch.mean(feat[labels==0], dim=0, keepdim=True)
                visualize(args, feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(), centers.data.cpu().numpy(),
                          epoch_num + 1, "Dev")

        if args.test_on_eval:
            with torch.no_grad():
                ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
                for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
                    # if i > 2: break
                    cqcc = cqcc.transpose(2,3).to(args.device)
                    tags = tags.to(args.device)
                    labels = labels.to(args.device)

                    feats, cqcc_outputs = cqcc_model(cqcc)

                    if args.base_loss == "bce":
                        cqcc_loss = criterion(cqcc_outputs, labels.unsqueeze(1).float())
                        score = cqcc_outputs[:, 0]
                    else:
                        cqcc_loss = criterion(cqcc_outputs, labels)
                        score = F.softmax(cqcc_outputs, dim=1)[:, 0]

                    ip1_loader.append(feats)
                    idx_loader.append((labels))
                    tag_loader.append((tags))

                    if args.add_loss in [None]:
                        testlossDict["base_loss"].append(cqcc_loss.item())
                    elif args.add_loss in ["lgm", "center"]:
                        testlossDict[args.add_loss].append(cqcc_loss.item())
                    elif args.add_loss == "lgcl":
                        outputs, moutputs = lgcl_loss(feats, labels)
                        cqcc_loss = criterion(moutputs, labels)
                        score = F.softmax(outputs, dim=1)[:, 0]
                        testlossDict[args.add_loss].append(cqcc_loss.item())
                    elif args.add_loss in ["isolate", "iso_sq"]:
                        isoloss = iso_loss(feats, labels)
                        score = torch.norm(feats - iso_loss.center, p=2, dim=1)
                        testlossDict[args.add_loss].append(isoloss.item())
                    elif args.add_loss == "ang_iso":
                        ang_isoloss, score = ang_iso(feats, labels)
                        testlossDict[args.add_loss].append(ang_isoloss.item())
                    elif args.add_loss == "multi_isolate":
                        multi_isoloss = multi_iso_loss(feats, labels)
                        testlossDict[args.add_loss].append(multi_isoloss.item())
                    elif args.add_loss == "multicenter_isolate":
                        multiisoloss = multicenter_iso_loss(feats, labels)
                        testlossDict[args.add_loss].append(multiisoloss.item())

                    score_loader.append(score)

                    # desc_str = ''
                    # for key in sorted(testlossDict.keys()):
                    #     desc_str += key + ':%.5f' % (np.nanmean(testlossDict[key])) + ', '
                    # # v.set_description(desc_str)
                    # print(desc_str)
                scores = torch.cat(score_loader, 0).data.cpu().numpy()
                labels = torch.cat(idx_loader, 0).data.cpu().numpy()
                eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
                other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
                eer = min(eer, other_eer)

                with open(os.path.join(args.out_fold, "test_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(np.nanmean(testlossDict[monitor_loss])) + "\t" + str(eer) + "\n")
                print("Test EER: {}".format(eer))


        valLoss = np.nanmean(devlossDict[monitor_loss])
        # if args.add_loss == "isolate":
        #     print("isolate center: ", iso_loss.center.data)
        if (epoch_num + 1) % 1 == 0:
            torch.save(cqcc_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_cqcc_model_%d.pt' % (epoch_num + 1)))
            if args.add_loss == "center":
                loss_model = centerLoss
                torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                    'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
            elif args.add_loss in ["isolate", "iso_sq"]:
                loss_model = iso_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                    'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
            elif args.add_loss == "ang_iso":
                loss_model = ang_iso
                torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                    'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
            elif args.add_loss == "lgm":
                loss_model = lgm_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                    'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
            elif args.add_loss == "lgcl":
                loss_model = lgcl_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                    'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
            else:
                loss_model = None

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
            if args.add_loss in ["isolate", "iso_sq"]:
                loss_model = iso_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            elif args.add_loss == "ang_iso":
                loss_model = ang_iso
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            elif args.add_loss == "lgm":
                loss_model = lgm_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            elif args.add_loss == "lgcl":
                loss_model = lgcl_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None
            prev_loss = valLoss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 500:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 499))
            break
        # if early_stop_cnt == 1:
        #     torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt')

            # print('Dev Accuracy of the model on the val features: {} % '.format(100 * cqcc_correct / total))

    return cqcc_model, loss_model



if __name__ == "__main__":
    args = initParams()
    if not args.test_only:
        _, _ = train(args)
    # model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
    # if args.add_loss is None:
    #     loss_model = None
    # else:
    #     loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
    # # TReer_cm, TRmin_tDCF = test(args, model, loss_model, "train")
    # # VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")
    # TEeer_cm, TEmin_tDCF = test(args, model, loss_model)
    # with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
    #     # res_file.write('\nTrain EER: %8.5f min-tDCF: %8.5f\n' % (TReer_cm, TRmin_tDCF))
    #     # res_file.write('\nVal EER: %8.5f min-tDCF: %8.5f\n' % (VAeer_cm, VAmin_tDCF))
    #     res_file.write('\nTest EER: %8.5f min-tDCF: %8.5f\n' % (TEeer_cm, TEmin_tDCF))
    # plot_loss(args)

    # # Test a checkpoint model
    # args = initParams()
    # model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_cqcc_model_19.pt'))
    # loss_model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_19.pt'))
    # VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")