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
from typing import Tuple
import soundfile as sf


torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    try:
        wave, sr = librosa.load(filepath, sr=16000)
    except:
        print(filepath)
        wave, sr = sf.read(filepath)
        print(sr == 16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


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
        filename = "_".join(filebasename.split("_")[:-1])

        speaker, tag, label = self.all_info[filename]

        return waveform, filename, tag, label, channel


class ASVspoof2019LARaw_withTransmissionAndDevice(Dataset):
    def __init__(self, path_to_database="/data/shared/LAPA_aug", path_to_protocol="/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part='train'):
        super(ASVspoof2019LARaw_withTransmissionAndDevice, self).__init__()
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
        channel = filebasename.split("_")[-2]
        device = filebasename.split("_")[-1]
        filename = "_".join(filebasename.split("_")[:-2])

        speaker, tag, label = self.all_info[filename]

        return waveform, filename, tag, label, channel, device


class ASVspoof2019DFRaw_withCompression(Dataset):
    def __init__(self, path_to_database="/data/shared/DF_aug", path_to_protocol="/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part='train'):
        super(ASVspoof2019DFRaw_withCompression, self).__init__()
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
        filename = "_".join(filebasename.split("_")[:-1])

        speaker, tag, label = self.all_info[filename]

        return waveform, filename, tag, label, channel


class ASVspoof2019DFRaw_withCompressionAndDevice(Dataset):
    def __init__(self, path_to_database="/data/shared/DFPA_aug", path_to_protocol="/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part='train'):
        super(ASVspoof2019DFRaw_withCompressionAndDevice, self).__init__()
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
        channel = filebasename.split("_")[-2]
        device = filebasename.split("_")[-1]
        filename = "_".join(filebasename.split("_")[:-2])

        speaker, tag, label = self.all_info[filename]

        return waveform, filename, tag, label, channel, device

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

    # asvspoof2021Raw_LA_aug = ASVspoof2019LARaw_withTransmission(part="train")
    # print(len(asvspoof2021Raw_LA_aug))
    # waveform, filename, tag, label, channel = asvspoof2021Raw_LA_aug[1230]
    # print(waveform.shape)
    # print(filename)
    # print(tag)
    # print(label)
    # print(channel)

    asvspoof2021Raw_LAPA_aug = ASVspoof2019LARaw_withTransmissionAndDevice(part="dev")
    print(len(asvspoof2021Raw_LAPA_aug))
    waveform, filename, tag, label, channel, device = asvspoof2021Raw_LAPA_aug[1230]
    print(waveform.shape)
    print(filename)
    print(tag)
    print(label)
    print(channel)
    print(device)
    device_lst = []
    for i in range(23423, 25599):
        waveform, filename, tag, label, channel, device = asvspoof2021Raw_LAPA_aug[i]
        if device not in device_lst:
            device_lst.append(device)
    print(device_lst)
