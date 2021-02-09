import logging
import os
import tarfile
import zipfile

from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
import urllib
import urllib.request
from torch.utils.model_zoo import tqdm
import random
import numpy as np
from dataset import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import eval_metrics as em
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
import time
from distutils import util

## Adapted from https://github.com/pytorch/audio/tree/master/torchaudio
## https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/

def str2bool(v):
    return bool(util.strtobool(v))

def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # # initialization
    # torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(random_seed)
    #     torch.backends.cudnn.deterministic = cudnn_deterministic
    #     torch.backends.cudnn.benchmark = False


def stream_url(url: str,
               start_byte: Optional[int] = None,
               block_size: int = 32 * 1024,
               progress_bar: bool = True) -> Iterable:
    """Stream url by chunk
    Args:
        url (str): Url.
        start_byte (int, optional): Start streaming at that point (Default: ``None``).
        block_size (int, optional): Size of chunks to stream (Default: ``32 * 1024``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
    """

    # If we already have the whole file, there is no need to download it again
    req = urllib.request.Request(url, method="HEAD")
    url_size = int(urllib.request.urlopen(req).info().get("Content-Length", -1))
    if url_size == start_byte:
        return

    req = urllib.request.Request(url)
    if start_byte:
        req.headers["Range"] = "bytes={}-".format(start_byte)

    with urllib.request.urlopen(req) as upointer, tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=url_size,
        disable=not progress_bar,
    ) as pbar:

        num_bytes = 0
        while True:
            chunk = upointer.read(block_size)
            if not chunk:
                break
            yield chunk
            num_bytes += len(chunk)
            pbar.update(len(chunk))

def download_url(url: str,
                 download_folder: str,
                 filename: Optional[str] = None,
                 hash_value: Optional[str] = None,
                 hash_type: str = "sha256",
                 progress_bar: bool = True,
                 resume: bool = False) -> None:
    """Download file to disk.
    Args:
        url (str): Url.
        download_folder (str): Folder to download file.
        filename (str, optional): Name of downloaded file. If None, it is inferred from the url (Default: ``None``).
        hash_value (str, optional): Hash for url (Default: ``None``).
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
        resume (bool, optional): Enable resuming download (Default: ``False``).
    """

    req = urllib.request.Request(url, method="HEAD")
    req_info = urllib.request.urlopen(req).info()

    # Detect filename
    filename = filename or req_info.get_filename() or os.path.basename(url)
    filepath = os.path.join(download_folder, filename)
    if resume and os.path.exists(filepath):
        mode = "ab"
        local_size: Optional[int] = os.path.getsize(filepath)

    elif not resume and os.path.exists(filepath):
        raise RuntimeError(
            "{} already exists. Delete the file manually and retry.".format(filepath)
        )
    else:
        mode = "wb"
        local_size = None

    if hash_value and local_size == int(req_info.get("Content-Length", -1)):
        with open(filepath, "rb") as file_obj:
            if validate_file(file_obj, hash_value, hash_type):
                return
        raise RuntimeError(
            "The hash of {} does not match. Delete the file manually and retry.".format(
                filepath
            )
        )

    with open(filepath, mode) as fpointer:
        for chunk in stream_url(url, start_byte=local_size, progress_bar=progress_bar):
            fpointer.write(chunk)

    with open(filepath, "rb") as file_obj:
        if hash_value and not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError(
                "The hash of {} does not match. Delete the file manually and retry.".format(
                    filepath
                )
            )

def extract_archive(from_path: str, to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    """Extract archive.
    Args:
        from_path (str): the path of the archive.
        to_path (str, optional): the root path of the extraced files (directory of from_path) (Default: ``None``)
        overwrite (bool, optional): overwrite existing files (Default: ``False``)
    Returns:
        list: List of paths to extracted files even if not overwritten.
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    try:
        with tarfile.open(from_path, "r") as tar:
            logging.info("Opened tar file {}.".format(from_path))
            files = []
            for file_ in tar:  # type: Any
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            return files
    except tarfile.ReadError:
        pass

    try:
        with zipfile.ZipFile(from_path, "r") as zfile:
            logging.info("Opened zip file {}.".format(from_path))
            files = zfile.namelist()
            for file_ in files:
                file_path = os.path.join(to_path, file_)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        return files
    except zipfile.BadZipFile:
        pass

    raise NotImplementedError("We currently only support tar.gz, tgz, and zip achives.")

def walk_files(root: str,
               suffix: Union[str, Tuple[str]],
               prefix: bool = False,
               remove_suffix: bool = False) -> Iterable[str]:
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the full path to each result, otherwise
            only returns the name of the files found (Default: ``False``)
        remove_suffix (bool, optional): If true, removes the suffix to each result defined in suffix,
            otherwise will return the result as found (Default: ``False``).
    """

    root = os.path.expanduser(root)

    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        # `dirs` is the list used in os.walk function and by sorting it in-place here, we change the
        # behavior of os.walk to traverse sub directory alphabetically
        # see also
        # https://stackoverflow.com/questions/6670029/can-i-force-python3s-os-walk-to-visit-directories-in-alphabetical-order-how#comment71993866_6670926
        files.sort()
        for f in files:
            if f.endswith(suffix):

                if remove_suffix:
                    f = f[: -len(suffix)]

                if prefix:
                    f = os.path.join(dirpath, f)

                yield f

def test_model(feat_model_path, loss_model_path, part, add_loss, add_external_genuine=False):
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
    test_set = ASVspoof2019("LA", "/dataNVME/neil/ASVspoof2019LA/", part,
                            "LFCC", feat_len=750, padding="repeat")
    # test_set_VCC = VCC2020("/data2/neil/VCC2020/", "LFCC", feat_len=750, padding="repeat")
    if add_external_genuine:
        external_genuine = LibriGenuine("/dataNVME/neil/libriTTS/train-clean-100", part="train", feature="LFCC", feat_len=750, padding="repeat")

        test_set += external_genuine
    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (lfcc, tags, labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
            # print(lfcc.shape)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]
            # print(score)

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    'A%02d %s %s\n' % (tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer

def test_model_on_PA(feat_model_path, loss_model_path, part, add_loss, add_external_genuine=False):
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
    test_set = ASVspoof2019("PA", "/data2/neil/ASVspoof2019PA/", part,
                            "LFCC", feat_len=750, padding="repeat")
    test_set_LA_genuine = ASVspoof2019("LA", "/data2/neil/ASVspoof2019LA/", part,
                            "LFCC", feat_len=750, padding="repeat", genuine_only=False)

    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    testLADataLoader = DataLoader(test_set_LA_genuine, batch_size=8, shuffle=False, num_workers=0)

    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
    #     for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
    #         lfcc = lfcc.transpose(2,3).to(device)
    #         # print(lfcc.shape)
    #         tags = tags.to(device)
    #         labels = labels.to(device)
    #
    #         feats, lfcc_outputs = model(lfcc)
    #
    #         score = F.softmax(lfcc_outputs)[:, 0]
    #         # print(score)
    #
    #         if add_loss == "ocsoftmax":
    #             ang_isoloss, score = loss_model(feats, labels)
    #         elif add_loss == "amsoftmax":
    #             outputs, moutputs = loss_model(feats, labels)
    #             score = F.softmax(outputs, dim=1)[:, 0]
    #         else: pass
    #
    #         for j in range(labels.size(0)):
    #             # if labels[j].data.cpu().numpy():
    #             cm_score_file.write(
    #                 '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
    #                                       "spoof" if labels[j].data.cpu().numpy() else "bonafide",
    #                                       score[j].item()))
    #
    #         score_loader.append(score.detach().cpu())
    #         idx_loader.append(labels.detach().cpu())

        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testLADataLoader)):
            lfcc = lfcc.transpose(2, 3).to(device)
            # print(lfcc.shape)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]
            # print(score)

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else:
                pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer


def test(model_dir, add_loss):
    model_path = os.path.join(model_dir, "anti-spoofing_cqcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    test_model(model_path, loss_model_path, "eval", add_loss)

def test_on_VCC(feat_model_path, loss_model_path, part, add_loss, add_external_genuine=False):
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
    test_set_VCC = VCC2020("/data2/neil/VCC2020/", "LFCC", feat_len=750, padding="repeat")
    testDataLoader = DataLoader(test_set_VCC, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_VCC.txt'), 'w') as cm_score_file:
        for i, (lfcc, tags, labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
            # print(lfcc.shape)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]
            # print(score)

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    'A%02d %s %s\n' % (tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer

def test_on_ASVspoof2015(feat_model_path, loss_model_path, part, add_loss, add_external_genuine=False):
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
    test_set_2015 = ASVspoof2015("/data2/neil/ASVspoof2015/", part="eval", feature="LFCC", feat_len=750, padding="repeat")
    print(len(test_set_2015))
    testDataLoader = DataLoader(test_set_2015, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_VCC.txt'), 'w') as cm_score_file:
        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]
            # print(score)

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer

def test_individual_attacks(cm_score_file):
    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(1, 12):
        # Extract target, nontarget, and spoof scores from the ASV scores

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        eer_cm_lst.append(min(eer_cm, other_eer_cm))

    return eer_cm_lst


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    # start = time.time()
    model_dir = "/data/neil/antiRes/models1028/ocsoftmax"
    # model_dir = "/data/neil/analyse/models0131/softmax"
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    # eer = test_model(model_path, loss_model_path, "eval", None, add_external_genuine=False)
    # eer = test_on_VCC(model_path, loss_model_path, "eval", "ocsoftmax", add_external_genuine=False)
    # eer = test_model_on_PA(model_path, loss_model_path, "eval", None, add_external_genuine=False)
    eer = test_on_ASVspoof2015(model_path, loss_model_path, "eval", "ocsoftmax", add_external_genuine=False)
    print(eer)
    eer_cm_lst = test_individual_attacks(os.path.join(model_dir, 'checkpoint_cm_score_VCC.txt'))
    print(eer_cm_lst)
    # print(time.time() - start)
    # eer = test_model(model_path, loss_model_path, "eval", "ocsoftmax", add_external_genuine=True)
    # print(eer)
