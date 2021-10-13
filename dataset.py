#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from feature_extraction import LFCC
from torch.utils.data.dataloader import default_collate

lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
wavform = torch.Tensor(np.expand_dims([0]*3200, axis=0))
lfcc_silence = lfcc(wavform)
silence_pad_value = lfcc_silence[:,0,:].unsqueeze(0)

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 6
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)


class ASVspoof2021LA_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/", path_to_augFeatures="/dataNVME/neil/ASVspoof2019LA_augFeatures", part="train", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LA_aug, self).__init__()
        if feature == "Melspec":
            path_to_augFeatures = "/data3/neil/ASVspoof2019LA_augFeatures"
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'amr[br=10k2,nodtx]', 'amr[br=5k9]', 'amr[br=6k7,nodtx]',
                        'amr[br=7k95,nodtx]', 'amrwb[br=12k65]', 'amrwb[br=15k85]', 'g711[law=a]',
                        'g711[law=u]', 'g722[br=64k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]',
                        'g726[law=u,40k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g728',
                        'silk[br=10k,loss=10]', 'silk[br=15k,loss=5]', 'silk[br=15k]',
                        'silk[br=20k,loss=5]', 'silk[br=5k,loss=10]', 'silk[br=5k]', 'amr[br=12k2]',
                        'amr[br=5k9,nodtx]', 'amrwb[br=6k6,nodtx]', 'g722[br=56k]', 'g726[law=a,br=32k]',
                        'g726[law=a,br=40k]', 'silk[br=15k,loss=10]', 'silk[br=20k]',
                        'silkwb[br=10k,loss=5]', 'amr[br=10k2]', 'amr[br=4k75]', 'amr[br=7k95]',
                        'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05]', 'g726[law=u,br=16k]', 'g729a',
                        'gsmfr', 'silkwb[br=10k,loss=10]', 'silkwb[br=20k]', 'silkwb[br=30k,loss=10]',
                        'amr[br=7k4,nodtx]', 'amrwb[br=6k6]', 'silk[br=10k]', 'silk[br=5k,loss=5]',
                        'silkwb[br=30k,loss=5]', 'amr[br=4k75,nodtx]', 'amr[br=7k4]', 'g722[br=48k]',
                        'silk[br=20k,loss=10]', 'silkwb[br=30k]', 'amr[br=5k15]',
                        'silkwb[br=20k,loss=5]', 'amrwb[br=23k05,nodtx]', 'amrwb[br=12k65,nodtx]',
                        'silkwb[br=20k,loss=10]', 'amr[br=6k7]', 'silkwb[br=10k]', 'silk[br=10k,loss=5]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 7
            channel = all_info[6]
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        #print(featureTensor.size())
        
        this_feat_len = featureTensor.shape[1]
        #print(this_feat_len)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
                
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, self.channel_dict[channel]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021LAPA_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/",
                 path_to_augFeatures="/data3/neil/ASVspoof2019LAPA_augFeatures", part="train", feature='LFCC',
                 feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LAPA_aug, self).__init__()
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'amr[br=10k2,nodtx]', 'amr[br=5k9]', 'amr[br=6k7,nodtx]',
                        'amr[br=7k95,nodtx]', 'amrwb[br=12k65]', 'amrwb[br=15k85]', 'g711[law=a]',
                        'g711[law=u]', 'g722[br=64k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]',
                        'g726[law=u,40k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g728',
                        'silk[br=10k,loss=10]', 'silk[br=15k,loss=5]', 'silk[br=15k]',
                        'silk[br=20k,loss=5]', 'silk[br=5k,loss=10]', 'silk[br=5k]', 'amr[br=12k2]',
                        'amr[br=5k9,nodtx]', 'amrwb[br=6k6,nodtx]', 'g722[br=56k]', 'g726[law=a,br=32k]',
                        'g726[law=a,br=40k]', 'silk[br=15k,loss=10]', 'silk[br=20k]',
                        'silkwb[br=10k,loss=5]', 'amr[br=10k2]', 'amr[br=4k75]', 'amr[br=7k95]',
                        'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05]', 'g726[law=u,br=16k]', 'g729a',
                        'gsmfr', 'silkwb[br=10k,loss=10]', 'silkwb[br=20k]', 'silkwb[br=30k,loss=10]',
                        'amr[br=7k4,nodtx]', 'amrwb[br=6k6]', 'silk[br=10k]', 'silk[br=5k,loss=5]',
                        'silkwb[br=30k,loss=5]', 'amr[br=4k75,nodtx]', 'amr[br=7k4]', 'g722[br=48k]',
                        'silk[br=20k,loss=10]', 'silkwb[br=30k]', 'amr[br=5k15]',
                        'silkwb[br=20k,loss=5]', 'amrwb[br=23k05,nodtx]', 'amrwb[br=12k65,nodtx]',
                        'silkwb[br=20k,loss=10]', 'amr[br=6k7]', 'silkwb[br=10k]', 'silk[br=10k,loss=5]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
        self.devices = ['OktavaML19-16000.ir', 'iPhoneirRecording-16000.ir', 'iPadirRecording-16000.ir',
                       'ResloRB250-16000.ir', 'telephonehornT65C-16000.ir', 'ResloSR1-16000.ir', 'RCAPB90-16000.ir',
                       'ResloRBRedLabel-16000.ir', 'telephone90sC-16000.ir', 'SonyC37Fet-16000.ir', 'Doremi-16000.ir',
                       'BehritoneirRecording-16000.ir', ""]
        self.device_dict = dict(zip(iter(self.devices), range(len(self.devices))))

    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
            device = ""
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename[:-3].split("_")
            assert len(all_info) == 8
            channel = all_info[6]
            device = all_info[7]
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        # print(featureTensor.size())

        this_feat_len = featureTensor.shape[1]
        # print(this_feat_len)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]

            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, \
               np.array([self.channel_dict[channel], self.device_dict[device]])

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021LAeval(Dataset):
    def __init__(self, path_to_features="/dataNVME/neil/ASVspoof2021LAFeatures", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LAeval, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 4
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:])
        return featureTensor, filename

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021DF_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/", path_to_augFeatures="/dataNVME/neil/ASVspoof2019DF_augFeatures", part="train", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DF_aug, self).__init__()
        if feature == "Melspec":
            path_to_augFeatures = "/data3/neil/ASVspoof2019DF_augFeatures"
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'aac[16k]', 'aac[32k]', 'aac[8k]', 'mp3[16k]', 'mp3[32k]', 'mp3[8k]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 7
            channel = all_info[6]
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, self.channel_dict[channel]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021DFPA_aug(Dataset):
    def __init__(self, path_to_ori="/data2/neil/ASVspoof2019LA/",
                 path_to_augFeatures="/data3/neil/ASVspoof2019DFPA_augFeatures", part="train", feature='LFCC',
                 feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DFPA_aug, self).__init__()
        self.path_to_features = path_to_augFeatures
        self.part = part
        self.ori = os.path.join(path_to_ori, part)
        self.ptf = os.path.join(path_to_augFeatures, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.ori_files = librosa.util.find_files(os.path.join(self.ori, self.feature), ext="pt")
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['no_channel', 'aac[16k]', 'aac[32k]', 'aac[8k]', 'mp3[16k]', 'mp3[32k]', 'mp3[8k]']
        self.channel_dict = dict(zip(iter(self.channel), range(len(self.channel))))
        self.devices = ['OktavaML19-16000.ir', 'iPhoneirRecording-16000.ir', 'iPadirRecording-16000.ir',
                       'ResloRB250-16000.ir', 'telephonehornT65C-16000.ir', 'ResloSR1-16000.ir', 'RCAPB90-16000.ir',
                       'ResloRBRedLabel-16000.ir', 'telephone90sC-16000.ir', 'SonyC37Fet-16000.ir', 'Doremi-16000.ir',
                       'BehritoneirRecording-16000.ir', ""]
        self.device_dict = dict(zip(iter(self.devices), range(len(self.devices))))

    def __len__(self):
        return len(self.ori_files) + len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.ori_files):
            filepath = self.ori_files[idx]
            basename = os.path.basename(filepath)
            all_info = basename.split(".")[0].split("_")
            assert len(all_info) == 6
            channel = "no_channel"
            device = ""
        else:
            filepath = self.all_files[idx - len(self.ori_files)]
            basename = os.path.basename(filepath)
            all_info = basename[:-3].split("_")
            assert len(all_info) == 8
            channel = all_info[6]
            device = all_info[7]
        featureTensor = torch.load(filepath)
        if self.feature == "Melspec":
            featureTensor = torch.unsqueeze(featureTensor, 0)
            featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        # print(featureTensor.size())

        this_feat_len = featureTensor.shape[1]
        # print(this_feat_len)
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]

            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, \
               np.array([self.channel_dict[channel], self.device_dict[device]])

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2021DFeval(Dataset):
    def __init__(self, path_to_features="/dataNVME/neil/ASVspoof2021DFFeatures", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DFeval, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 4
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:])
        return featureTensor, filename

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


def padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros((1, padd_len, width), dtype=spec.dtype)), 1)

def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul, 1)[:, :ref_len, :]
    return spec

def silence_padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((silence_pad_value.repeat(1, padd_len, 1).to(spec.device), spec), 1)



if __name__ == "__main__":
    # path_to_features = '/data2/neil/ASVspoof2019LA/'  # if run on GPU
    # training_set = ASVspoof2019("LA", path_to_features, 'train',
    #                             'LFCC', feat_len=750, pad_chop=False, padding='repeat')
    # feat_mat, tag, label = training_set[2999]
    # print(len(training_set))
    # # print(this_len)
    # print(feat_mat.shape)
    # print(tag)
    # print(label)

    # samples = [training_set[26], training_set[27], training_set[28], training_set[29]]
    # out = training_set.collate_fn(samples)

    # training_set = ASVspoof2015("/data2/neil/ASVspoof2015/", part="eval")
    # feat_mat, _, tag, label = training_set[299]
    # print(len(training_set))
    # print(tag)
    # print(label)

    # training_set = ASVspoof2021LA_aug()
    # for i in range(25379, 26950):
    #     feat_mat, filename, tag, label, channel = training_set[i]
    # print(len(training_set))
    # print(filename)
    # print(feat_mat.shape)
    # print(tag)
    # print(label)
    # print(channel)
    # print(len(training_set.channel))
    # print(training_set.channel)


    # trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0, collate_fn=training_set.collate_fn)
    # feat_mat_batch, _, tags, labels = [d for d in next(iter(trainDataLoader))]
    # print(feat_mat_batch.shape)
    # print(this_len)
    # print(feat_mat_batch)


    # asvspoof = ASVspoof2019("LA", "/data2/neil/ASVspoof2019LA/", part='train', feature='LFCC', feat_len=750, padding='repeat', genuine_only=True)
    # print(len(asvspoof))
    # featTensor, tag, label = asvspoof[2579]
    # print(featTensor.shape)
    # # print(filename)
    # print(tag)
    # print(label)

    # channel_lst = []
    # training_set = ASVspoof2021DF_aug()
    # for i in range(25379, 26950):
    #     feat_mat, filename, tag, label, channel = training_set[i]
    #     if not channel in channel_lst:
    #         channel_lst.append(channel)
    # print(channel_lst)

    training_set = ASVspoof2021LAPA_aug()
    feat_mat, filename, tag, label, channel = training_set[12556]
    print(channel)
