from dataset import ASVspoof2021LAeval
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

def test_on_ASVspoof2021(feat_model_path, loss_model_path, part, add_loss, add_external_genuine=False):
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
    test_set = ASVspoof2021LAeval()
    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    model.eval()

    with open(os.path.join(dir_path, 'score.txt'), 'w') as cm_score_file:
        for i, (lfcc, audio_fn) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
            labels = torch.zeros((lfcc.shape[0]))

            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s %s\n' % (audio_fn[j], -score[j].item()))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")
    # model_dir = "/data/neil/antiRes/models1028/ocsoftmax"
    # model_dir = "/data/analyse/channel0321/aug"
    model_dir = "/data/analyse/channel0321/adv_0.001"
    model_path = os.path.join(model_dir, "anti-spoofing_cqcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    test_on_ASVspoof2021(model_path, loss_model_path, "eval", "ocsoftmax", add_external_genuine=False)
