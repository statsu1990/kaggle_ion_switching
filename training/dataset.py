import torch
import numpy as np

class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, signals, group, open_channels=None, transform=None, use_trans_matrix=True, prob_temperature=1.0):
        self.signals = signals
        self.group = group
        self.open_channels = open_channels if open_channels is not None else None
        self.transform = transform
        self.use_trans_matrix = use_trans_matrix
        self.prob_temperature = prob_temperature

    def __getitem__(self, idx):
        sig = self.signals[idx]
        gr = self.group[idx]
        op_chn = self.open_channels[idx] if self.open_channels is not None else None

        if self.transform is not None:
            sig, op_chn = self.transform(sig, op_chn)

        if self.open_channels is not None:
            if self.use_trans_matrix:
                tr_mt = _calc_trans_matrix(op_chn, max_open_channels=10, prob_temperature=self.prob_temperature)
                return sig, gr, op_chn, tr_mt
            else:
                return sig, gr, op_chn
        else:
            return sig, gr

    def __len__(self):
        return len(self.signals)

def get_dataloader(dataset, batch_size, shuffle=True, weights=None):
    if weights is None:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(dataset), replacement=True)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

def _calc_trans_matrix(open_channels, max_open_channels=10, prob_temperature=1.0):
    """
    refference : https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution

    Returns:
        trans matrix(i, j) : transition probability of state i to j
    """
    if type(open_channels) is torch.Tensor:
        op_chs = open_channels.detach().numpy().copy()
    else:
        op_chs = open_channels

    oc_next = np.roll(op_chs, -1)
    
    trans_mtrx = []
    
    for oc in range(max_open_channels + 1):
        trans_rate = np.histogram(oc_next[op_chs==oc], bins=np.arange(max_open_channels+2))[0]
        if np.sum(trans_rate) == 0:
            trans_rate = np.zeros_like(trans_rate)
        else:
            trans_rate = trans_rate / np.sum(trans_rate) # normalize to 1
            trans_rate = np.power(trans_rate, prob_temperature)
            trans_rate = trans_rate / np.sum(trans_rate) # normalize to 1

        trans_mtrx.append(trans_rate)

    if type(open_channels) is torch.Tensor:
        return torch.from_numpy(np.array(trans_mtrx).astype(np.float32)).clone()
    else:
        return np.array(trans_mtrx).astype(np.float32)
