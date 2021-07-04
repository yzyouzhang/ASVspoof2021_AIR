from dataset import *
from model import *
from ecapa_tdnn import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

def test_on_ASVspoof2021(task, feat_model_path, loss_model_path, output_score_path, part, add_loss):
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
    elif task == "19augdev":
        test_set = ASVspoof2021LA_aug(part="dev", pad_chop=True)
    else:
        print("what task?")
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    with open(os.path.join(output_score_path, 'score.txt'), 'w') as cm_score_file:
        for i, (lfcc, audio_fn) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
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

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s %s\n' % (audio_fn[j], -score[j].item()))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda")
    # model_dir = "/data/neil/antiRes/models1028/ocsoftmax"
    # model_dir = "/data/analyse/channel0321/aug"
    # model_dir = "/data/analyse/channel0321/adv_0.001"
    # model_dir = "/data/neil/asv2021/models0609/LFCC+LCNN+P2SGrad+LAaug"

    ## Things need to change
    model_name = "lfcc_ecapa1024cfst_p2s"
    task = "LA"
    loss_for_eval = "p2sgrad"

    model_dir = os.path.join("/data/xinhui/models/", model_name)
    output_score_path = os.path.join("/data/neil/scores", model_name+task)
    if not os.path.exists(output_score_path):
        os.makedirs(output_score_path)
    model_path = os.path.join(model_dir, "anti-spoofing_cqcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    test_on_ASVspoof2021(task, model_path, loss_model_path, output_score_path, "eval", loss_for_eval)
