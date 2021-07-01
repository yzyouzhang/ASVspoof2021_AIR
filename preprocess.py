import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda" if cuda else "cpu")

# vctk = dataset.VCTK_092(root="/data/neil/VCTK", download=False)
# target_dir = "/dataNVME/neil/VCTK/LFCC"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# for idx in range(len(vctk)):
#     print("Processing", idx)
#     waveform, sample_rate, utterance, speaker_id, utterance_id = vctk[idx]
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s.pt" %(idx, speaker_id, utterance_id)))
# print("Done!")

# librispeech = dataset.LIBRISPEECH(root="/data/neil")
# target_dir = "/dataNVME/neil/libriSpeech/LFCC"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(librispeech)):
#     print("Processing", idx)
#     waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = librispeech[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" %(idx, speaker_id, chapter_id, utterance_id)))
# print("Done!")

# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data/neil/DS_10283_3336/", "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/data2/neil/ASVspoof2019LA", part_, "STFT")
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     spec = STFT(320, 160, 512, 16000)
#     spec = spec.to(device)
#     for idx in tqdm(range(len(asvspoof_raw))):
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = waveform.to(device)
#         specOfWav = spec(waveform).float()
#         torch.save(specOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")

# libritts = dataset.LIBRITTS(root="/data/neil")
# print(len(libritts))
# target_dir = "/dataNVME/neil/libriTTS/train-clean-100/LFCC/"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(libritts)):
#     print("Processing", idx)
#     waveform, sample_rate, text, normed_text, speaker_id, chapter_id, utterance_id = libritts[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" %(idx, speaker_id, chapter_id, utterance_id)))
# print("Done!")

# vcc2020 = dataset.VCC2020Raw()
# print(len(vcc2020))
# target_dir = "/data2/neil/VCC2020/LFCC/"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(vcc2020)):
#     print("Processing", idx)
#     waveform, filename, tag, label = vcc2020[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%04d_%s_%s_%s.pt" %(idx, filename, tag, label)))
# print("Done!")

#for part_ in ["train", "dev", "eval"]:
#    asvspoof_raw = dataset.ASVspoof2015Raw("/data/neil/ASVspoof2015/wav", "/data/neil/ASVspoof2015/CM_protocol", part=part_)
#    target_dir = os.path.join("/data2/neil/ASVspoof2015", part_, "LFCC")
#    lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#    lfcc = lfcc.to(device)
#    for idx in range(len(asvspoof_raw)):
#        print("Processing", idx)
#        waveform, filename, tag, label = asvspoof_raw[idx]
#        waveform = waveform.to(device)
#        lfccOfWav = lfcc(waveform)
#        torch.save(lfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#    print("Done!")
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = waveform.to(device)
#         lfccOfWav = lfcc(waveform)
#         torch.save(lfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")

# libritts = dataset.LIBRITTS(root="/data/neil")
# print(len(libritts))
# target_dir = "/dataNVME/neil/libriTTS/train-clean-100/LFCC/"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(libritts)):
#     print("Processing", idx)
#     waveform, sample_rate, text, normed_text, speaker_id, chapter_id, utterance_id = libritts[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" %(idx, speaker_id, chapter_id, utterance_id)))
# print("Done!")

# vcc2020 = dataset.VCC2020Raw()
# print(len(vcc2020))
# target_dir = "/data2/neil/VCC2020/LFCC/"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(vcc2020)):
#     print("Processing", idx)
#     waveform, filename, tag, label = vcc2020[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%04d_%s_%s_%s.pt" %(idx, filename, tag, label)))
# print("Done!")

# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2015Raw("/data/neil/ASVspoof2015/wav", "/data/neil/ASVspoof2015/CM_protocol", part=part_)
#     target_dir = os.path.join("/data2/neil/ASVspoof2015", part_, "LFCC")
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in range(len(asvspoof_raw)):
#         print("Processing", idx)
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = waveform.to(device)
#         lfccOfWav = lfcc(waveform)
#         torch.save(lfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")


# asvspoof2021_raw = dataset.ASVspoof2021evalRaw("/data2/neil/ASVspoof2021/ASVspoof2021_DF_eval/flac")
# target_dir = os.path.join("/dataNVME/neil/ASVspoof2021DFFeatures", "LFCC")
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in tqdm(len(asvspoof2021_raw)):
# # for idx in tqdm(list(range(20503, 20505))+list(range(20500+64202, 20500+64204))+list(range(20500+481274, 20500+481276))):
#     waveform, filename = asvspoof2021_raw[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%06d_%s.pt" % (idx, filename)))
# print("Done!")


# for part_ in ["train", "dev"]:
#     asvspoof2021Raw_LA_aug = dataset.ASVspoof2019LARaw_withTransmission(part=part_)
#     target_dir = os.path.join("/dataNVME/neil/ASVspoof2019LA_augFeatures", part_, "LFCC")
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in tqdm(range(520193, len(asvspoof2021Raw_LA_aug))):
#         try:
#             waveform, filename, tag, label, channel = asvspoof2021Raw_LA_aug[idx]
#             waveform = waveform.to(device)
#             lfccOfWav = lfcc(waveform)
#             torch.save(lfccOfWav, os.path.join(target_dir, "%06d_%s_%s_%s_%s.pt" % (idx, filename, tag, label, channel)))
#         except:
#             print(idx)
#     print("Done!")

# for part_ in ["train", "dev"]:
#     asvspoof2021Raw_DF_aug = dataset.ASVspoof2019DFRaw_withCompression(part=part_)
#     target_dir = os.path.join("/dataNVME/neil/ASVspoof2019DF_augFeatures", part_, "LFCC")
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in tqdm(range(len(asvspoof2021Raw_DF_aug))):
#         try:
#             waveform, filename, tag, label, channel = asvspoof2021Raw_DF_aug[idx]
#             waveform = waveform.to(device)
#             lfccOfWav = lfcc(waveform)
#             torch.save(lfccOfWav, os.path.join(target_dir, "%06d_%s_%s_%s_%s.pt" % (idx, filename, tag, label, channel)))
#         except:
#             print(idx)
#     print("Done!")

for part_ in ["train", "dev"]:
    asvspoof2021Raw_LA_aug = dataset.ASVspoof2019LARaw_withTransmission(part=part_)
    target_dir = os.path.join("/dataNVME/neil/ASVspoof2019LA_augFeatures", part_, "Melspec")
    mel = Melspec()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for idx in tqdm(range(len(asvspoof2021Raw_LA_aug))):
        # try:
        waveform, filename, tag, label, channel = asvspoof2021Raw_LA_aug[idx]
        wav_mel = mel(waveform)
        torch.save(wav_mel, os.path.join(target_dir, "%06d_%s_%s_%s_%s.pt" % (idx, filename, tag, label, channel)))
        # except:
        #     print(idx)
    print("Done!")


# asvspoof2021_raw = dataset.ASVspoof2021evalRaw("/data2/neil/ASVspoof2021/ASVspoof2021_LA_eval/flac")
# target_dir = os.path.join("/dataNVME/neil/ASVspoof2021LAFeatures", "Melspec")
# mel = Melspec()
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)
# for idx in tqdm(range(len(asvspoof2021_raw))):
# # for idx in tqdm(list(range(20503, 20505))+list(range(20500+64202, 20500+64204))+list(range(20500+481274, 20500+481276))):
#     waveform, filename = asvspoof2021_raw[idx]
#     wav_mel = mel(waveform)
#     torch.save(wav_mel, os.path.join(target_dir, "%06d_%s.pt" % (idx, filename)))
# print("Done!")

# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data/neil/DS_10283_3336/", "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/data2/neil/ASVspoof2019LA", part_, "Melspec")
#     mel = Melspec()
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     for idx in tqdm(range(len(asvspoof_raw))):
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         # waveform = waveform.to(device)
#         wav_mel = mel(waveform)
#         torch.save(wav_mel.float(), os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")
