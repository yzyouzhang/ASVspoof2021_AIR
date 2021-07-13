from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import zipfile


def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='/data/xinhui/models/')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model",
                        required=True, default='lfcc_ecapa512ctst_ocs')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score",
                        default='/data/neil/scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                        required=True, default='LA', choices=["LA", "DF", "19dev",
                                                              "19laaugdev", "19lapaaugdev",
                                                              "19dfaugdev", "19dfpaaugdev", "19eval"])
    parser.add_argument('-l', '--loss', help='loss for scoring', default=None,
                        required=False, choices=[None, "ocsoftmax", "amsoftmax", "p2sgrad"])
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if '19' in args.task:
        args.out_score_dir = "./scores"
    else:
        args.out_score_dir = args.score_dir

    return args

def zip_txt_file(txt_file, zip_name):
    pass


def test_on_ASVspoof2021(task, feat_model_path, loss_model_path, output_score_path, model_name, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    # model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    loss_model = torch.load(loss_model_path) if add_loss is not None else None

    ### use this line to generate score for LA 2021 Challenge
    if task == "LA":
        test_set = ASVspoof2021LAeval(pad_chop=True)
    ### use this line to generate score for DF 2021 Challenge
    elif task == "DF":
        test_set = ASVspoof2021DFeval(pad_chop=True)
    ### use this one to tune the fusion weights on the original dev set
    elif task == "19dev":
        test_set = ASVspoof2019("LA", "/data2/neil/ASVspoof2019LA", 'dev', "LFCC", pad_chop=True)
    ### use this one to tune the fusion weights on the augmented dev set
    elif task == "19laaugdev":
        test_set = ASVspoof2021LA_aug(part="dev", pad_chop=True)
    elif task == "19lapaaugdev":
        test_set = ASVspoof2021LAPA_aug(part="dev", pad_chop=True)
    elif task == "19dfaugdev":
        test_set = ASVspoof2021DF_aug(part="dev", pad_chop=True)
    elif task == "19dfpaaugdev":
        test_set = ASVspoof2021DFPA_aug(part="dev", pad_chop=True)
    ### use this one to tune the fusion weights on the augmented dev set
    elif task == '19eval':
        test_set = ASVspoof2019("LA", "/data2/neil/ASVspoof2019LA", 'eval', "LFCC", pad_chop=True)
    else:
        print("what task?")
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    if '19' in task:
        txt_file_name = os.path.join(output_score_path, model_name + '_' + task + '_score.txt')
    else:
        txt_dir = os.path.join(output_score_path, model_name + '_' + task)
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        txt_file_name = os.path.join(txt_dir,  'score.txt')

    with open(txt_file_name, 'w') as cm_score_file:
        for i, data_slice in enumerate(tqdm(testDataLoader)):
            if '19' in task:
                lfcc, audio_fn, labels = data_slice[0], data_slice[1], data_slice[3]
            else:
                lfcc, audio_fn = data_slice
            
            if 'ecapa' in model_path:
                lfcc = lfcc.transpose(2, 3).squeeze(1).to(device)
            else:
                lfcc = lfcc.transpose(2, 3).to(device)
                
            labels = torch.zeros((lfcc.shape[0]))

            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = -F.softmax(lfcc_outputs)[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "p2sgrad":
                outputs, score = loss_model(feats, labels)
                # score = F.softmax(outputs, dim=1)[:, 0]
            else: pass
            
            if '19' in task:
                for j in range(labels.size(0)):
                    cm_score_file.write('%s %s %s\n' % (audio_fn[j], -score[j].item(), "spoof" if labels[j].data.cpu().numpy() else "bonafide"))
            else:
                for j in range(labels.size(0)):
                    cm_score_file.write('%s %s\n' % (audio_fn[j], -score[j].item()))

if __name__ == "__main__":
    args = init()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")
    #
    # task = "19eval"
    # model_folder = "/data/xinhui/models"
    # model_name = "lfcc_ecapa512ctst_ocs"

    model_dir = os.path.join(args.model_folder, args.model_name)
    # loss_for_eval = "ocsoftmax"
    # score_path = "./scores"

    model_path = os.path.join(model_dir, "anti-spoofing_cqcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    test_on_ASVspoof2021(args.task, model_path, loss_model_path, args.score_dir, args.model_name, args.loss)