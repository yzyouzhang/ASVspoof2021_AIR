import torch
import torch.utils.data as torch_data
import torch.utils.data.sampler as torch_sampler
from dataset import *
from tqdm import tqdm

class SamplerWithRatio(torch_sampler.Sampler):
    """
    contro how we sample from the dataset. There are two part: one is the original,
    and the other one is the augmented with channel effects.
    """
    def __init__(self, data_source, ratio=0.5):
        self.num_samples = len(data_source)
        # cqcc, audio_fn, tags, labels, channels = data_source
        # print(len(channels))
        self.generator = torch.Generator()
        self.generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))


    def __iter__(self):
        # print(iter(range(self.num_samples)))
        # return (range(self.num_samples))
        yield from torch.randperm(self.num_samples, generator=self.generator).tolist()
        # return (0)*64

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    training_set = ASVspoof2021LA_aug()
    # feat_mat, filename, tag, label, channel = training_set[345]
    trainDataLoader = DataLoader(training_set, batch_size=64,
                                 shuffle=False, num_workers=0, sampler=SamplerWithRatio(training_set, 0.5))
    for i, (cqcc, audio_fn, tags, labels, channels) in enumerate(tqdm(trainDataLoader)):
        print(list(channels))
        count = 0
        for channel in list(channels):
            if channel == "no_channel":
                count += 1
        print(count / 64)
        if i > 1: break

