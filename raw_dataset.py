#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
import warnings
from typing import Any, Tuple, Union
from pathlib import Path
from utils import download_url, extract_archive, walk_files


torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

def load_librispeech_item(fileid: str,
                          path: str,
                          ext_audio: str,
                          ext_txt: str) -> Tuple[Tensor, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio_load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return (
        waveform,
        sample_rate,
        utterance,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )

class LIBRISPEECH(Dataset):
    """Create a Dataset for LibriSpeech.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self,
                 root: Union[str, Path],
                 basename: str = "train-clean-360",
                 folder_in_archive: str = "LibriSpeech") -> None:

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*/*/*' + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)


def load_libritts_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
) -> Tuple[Tensor, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
    utterance_id = fileid

    normalized_text = utterance_id + ext_normalized_txt
    normalized_text = os.path.join(path, speaker_id, chapter_id, normalized_text)

    original_text = utterance_id + ext_original_txt
    original_text = os.path.join(path, speaker_id, chapter_id, original_text)

    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio_load(file_audio)

    # Load original text
    with open(original_text) as ft:
        original_text = ft.readline()

    # Load normalized text
    with open(normalized_text, "r") as ft:
        normalized_text = ft.readline()

    return (
        waveform,
        sample_rate,
        original_text,
        normalized_text,
        int(speaker_id),
        int(chapter_id),
        utterance_id,
    )


class LIBRITTS(Dataset):
    """Create a Dataset for LibriTTS.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriTTS"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_original_txt = ".original.txt"
    _ext_normalized_txt = ".normalized.txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
        url: str = "train-clean-100",
        folder_in_archive: str = "LibriTTS",
        download: bool = False) -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/60/"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        _CHECKSUMS = {
            "http://www.openslr.org/60/dev-clean.tar.gz": "0c3076c1e5245bb3f0af7d82087ee207",
            "http://www.openslr.org/60/dev-other.tar.gz": "815555d8d75995782ac3ccd7f047213d",
            "http://www.openslr.org/60/test-clean.tar.gz": "7bed3bdb047c4c197f1ad3bc412db59f",
            "http://www.openslr.org/60/test-other.tar.gz": "ae3258249472a13b5abef2a816f733e4",
            "http://www.openslr.org/60/train-clean-100.tar.gz": "4a8c202b78fe1bc0c47916a98f3a2ea8",
            "http://www.openslr.org/60/train-clean-360.tar.gz": "a84ef10ddade5fd25df69596a2767b2d",
            "http://www.openslr.org/60/train-other-500.tar.gz": "7b181dd5ace343a5f38427999684aa6f",
        }

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*/*/*' + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int, int, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, original_text, normalized_text, speaker_id,
            chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_libritts_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_original_txt,
            self._ext_normalized_txt,
        )

    def __len__(self) -> int:
        return len(self._walker)


class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
                                    '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class VCC2020Raw(Dataset):
    def __init__(self, path_to_spoof="/data2/neil/nii-yamagishilab-VCC2020-listeningtest-31f913c", path_to_bonafide="/data2/neil/nii-yamagishilab-VCC2020-database-0b2fb2e"):
        super(VCC2020Raw, self).__init__()
        self.all_spoof = librosa.util.find_files(path_to_spoof, ext="wav")
        self.all_bonafide = librosa.util.find_files(path_to_bonafide, ext="wav")

    def __len__(self):
        # print(len(self.all_spoof), len(self.all_bonafide))
        return len(self.all_spoof) + len(self.all_bonafide)

    def __getitem__(self, idx):
        if idx < len(self.all_bonafide):
            filepath = self.all_bonafide[idx]
            label = "bonafide"
            filename = "_".join(filepath.split("/")[-3:])[:-4]
            tag = "-"
        else:
            filepath = self.all_spoof[idx - len(self.all_bonafide)]
            filename = os.path.basename(filepath)[:-4]
            label = "spoof"
            tag = filepath.split("/")[-3]
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2015Raw(Dataset):
    def __init__(self, path_to_database="/data/neil/ASVspoof2015/wav", path_to_protocol="/data/neil/ASVspoof2015/CM_protocol", part='train'):
        super(ASVspoof2015Raw, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part)
        self.path_to_protocol = path_to_protocol
        cm_pro_dict = {"train": "cm_train.trn", "dev": "cm_develop.ndx", "eval": "cm_evaluation.ndx"}
        protocol = os.path.join(self.path_to_protocol, cm_pro_dict[self.part])
        self.tag = {"human": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
                    "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10}
        self.label = {"spoof": 1, "human": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, speaker, filename + ".wav")
        waveform, sr = torchaudio_load(filepath)
        filename = filename.replace("_", "-")
        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2021evalRaw(Dataset):
    def __init__(self, path_to_database="/data2/neil/ASVspoof2021/ASVspoof2021_LA_eval/flac"):
        super(ASVspoof2021evalRaw, self).__init__()
        self.ptd = path_to_database
        self.path_to_audio = self.ptd
        self.all_files = librosa.util.find_files(self.path_to_audio, ext="flac")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        waveform, sr = torchaudio_load(filepath)
        filename = os.path.basename(filepath)[:-5]
        return waveform, filename


class ASVspoof2019LARaw_withTransmission(Dataset):
    def __init__(self, path_to_database="/data/shared/LA_aug", path_to_protocol="/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part='train'):
        super(ASVspoof2019LARaw_withTransmission, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part)
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(self.path_to_protocol,
                                'ASVspoof2019.' + "LA" + '.cm.' + self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, "LA", 'ASVspoof2019_' + "LA" +
                                    '_cm_protocols/ASVspoof2019.' + "LA" + '.cm.' + self.part + '.trl.txt')
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7}
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(self.path_to_audio, ext="wav")

        with open(protocol, 'r') as f:
            audio_info = {}
            for info in f.readlines():
                speaker, filename, _, tag, label = info.strip().split()
                audio_info[filename] = (speaker, tag, label)
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        waveform, sr = torchaudio_load(filepath)
        filebasename = os.path.basename(filepath)[:-4]
        channel = filebasename.split("_")[-1]
        filename = filebasename[:12]
        speaker, tag, label = self.all_info[filename]

        return waveform, filename, tag, label, channel


if __name__ == "__main__":
    # vctk = VCTK_092(root="/data/neil/VCTK", download=False)
    # print(len(vctk))
    # waveform, sample_rate, utterance, speaker_id, utterance_id = vctk[124]
    # print(waveform.shape)
    # print(sample_rate)
    # print(utterance)
    # print(speaker_id)
    # print(utterance_id)
    #
    # librispeech = LIBRISPEECH(root="/data/neil")
    # print(len(librispeech))
    # waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = librispeech[164]
    # print(waveform.shape)
    # print(sample_rate)
    # print(utterance)
    # print(speaker_id)
    # print(chapter_id)
    # print(utterance_id)
    #
    # libriGen = LibriGenuine("/dataNVME/neil/libriSpeech/", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat')
    # print(len(libriGen))
    # featTensor, tag, label = libriGen[123]
    # print(featTensor.shape)
    # print(tag)
    # print(label)
    #
    # asvspoof_raw = ASVspoof2019Raw("LA", "/data/neil/DS_10283_3336/", "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part="eval")
    # print(len(asvspoof_raw))
    # waveform, filename, tag, label = asvspoof_raw[123]
    # print(waveform.shape)
    # print(filename)
    # print(tag)
    # print(label)

    # vcc2020_raw = VCC2020Raw()
    # print(len(vcc2020_raw))
    # waveform, filename, tag, label = vcc2020_raw[123]
    # print(waveform.shape)
    # print(filename)
    # print(tag)
    # print(label)

    # asvspoof2015 = ASVspoof2015Raw(part="eval")
    # print(len(asvspoof2015))
    # waveform, filename, tag, label = asvspoof2015[123]
    # print(waveform.shape)
    # print(filename)
    # print(tag)
    # print(label)
    # pass

    asvspoof2021Raw_LA_aug = ASVspoof2019LARaw_withTransmission(part="train")
    print(len(asvspoof2021Raw_LA_aug))
    waveform, filename, tag, label, channel = asvspoof2021Raw_LA_aug[1230]
    print(waveform.shape)
    print(filename)
    print(tag)
    print(label)
    print(channel)
