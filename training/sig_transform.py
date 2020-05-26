import numpy as np
import math

class Compose:
    def __init__(self, transformer_list):
        self.transformers = transformer_list

    def __call__(self, signal, open_channel):
        trans_sig = signal
        trans_op_chn = open_channel

        for trns in self.transformers:
            trans_sig, trans_op_chn = trns(trans_sig, trans_op_chn)

        return trans_sig, trans_op_chn

class Normlize:
    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale

    def __call__(self, signal, open_channel=None):
        trans_sig = (signal - self.shift) / self.scale

        return trans_sig, open_channel

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal, open_channel):
        if np.random.rand() <= self.p:
            trans_sig = np.flip(signal, axis=0).copy()
            if open_channel is not None:
                trans_op_chn = np.flip(open_channel, axis=0).copy()
            else:
                trans_op_chn = open_channel
        else:
            trans_sig = signal
            trans_op_chn = open_channel

        return trans_sig, trans_op_chn

class RandomCrop:
    def __init__(self, crop_length, p=0.5):
        self.crop_length = crop_length
        self.p = p

    def __call__(self, signal, open_channel):
        if np.random.rand() <= self.p:
            diff_len = len(signal) - self.crop_length

            if diff_len > 0:
                start_idx = np.random.randint(0, diff_len+1)
                trans_sig = signal[start_idx:start_idx+self.crop_length]
                trans_op_chn = open_channel[start_idx:start_idx+self.crop_length]
            else:
                trans_sig = signal
                trans_op_chn = open_channel
        else:
            trans_sig = signal
            trans_op_chn = open_channel

        return trans_sig, trans_op_chn

class Cutout:
    def __init__(self, num_holes=5, max_size=1, fill_value=0, p=0.5):
        self.num_holes = num_holes
        self.max_size = max_size
        self.fill_value = fill_value
        self.p = p

        return

    def __call__(self, signal, open_channel):
        if np.random.rand() <= self.p:
            length = len(signal)
            hole_size = np.random.randint(1, self.max_size+1) if self.max_size > 1 else 1            
            hole_idxs = np.random.randint(0, length, self.num_holes)

            trans_sig = signal.copy()
            for i in range(hole_size):
                trans_sig[np.minimum(hole_idxs+i, length-1)] = self.fill_value

            trans_op_chn = open_channel
        else:
            trans_sig = signal
            trans_op_chn = open_channel

        return trans_sig, trans_op_chn

class GaussNoise:
    def __init__(self, std_limit=(0.05, 0.15), mean=0, p=0.5):
        self.std_limit = std_limit
        self.mean = mean
        self.p = p

    def __call__(self, signal, open_channel):
        if np.random.rand() <= self.p:
            std = self.std_limit[0] + np.random.rand() * (self.std_limit[1] - self.std_limit[0])
            noise = np.random.normal(self.mean, std, signal.shape).astype('float32')
            trans_sig = signal + noise
            trans_op_chn = open_channel
        else:
            trans_sig = signal
            trans_op_chn = open_channel

        return trans_sig, trans_op_chn

class RandomGain:
    def __init__(self, gain_limit=(0.9, 1.1), p=0.5):
        self.gain_limit = gain_limit
        self.p = p

    def __call__(self, signal, open_channel):
        if np.random.rand() <= self.p:
            gain = self.gain_limit[0] + np.random.rand() * (self.gain_limit[1] - self.gain_limit[0])
            trans_sig = signal * gain
            trans_op_chn = open_channel
        else:
            trans_sig = signal
            trans_op_chn = open_channel

        return trans_sig, trans_op_chn

class RandomDrift:
    def __init__(self, drift_a0_limit=(-0.3, 0.3), drift_a1_limit=(-0.4/1000, 0.4/1000), p=0.5, target_idxs=None):
        self.drift_a0_limit = drift_a0_limit
        self.drift_a1_limit = drift_a1_limit
        self.p = p
        self.target_idxs = np.array(target_idxs)

    def __call__(self, signal, open_channel):
        if np.random.rand() <= self.p:
            drift_a0 = self.drift_a0_limit[0] + np.random.rand() * (self.drift_a0_limit[1] - self.drift_a0_limit[0])
            drift_a1 = self.drift_a1_limit[0] + np.random.rand() * (self.drift_a1_limit[1] - self.drift_a1_limit[0])
            
            if self.target_idxs is None:
                trans_sig = signal + drift
            else:
                trans_sig = signal.copy()
                trans_sig[:,self.target_idxs] = signal[:,self.target_idxs] + drift_a0 + np.linspace(-len(trans_sig)*drift_a1/2, len(trans_sig)*drift_a1/2, len(trans_sig))[:,None]
            
            trans_op_chn = open_channel
        else:
            trans_sig = signal
            trans_op_chn = open_channel

        return trans_sig, trans_op_chn

class AxesTranspose:
    def __init__(self, axes=(1,0)):
        self.axes = axes

    def __call__(self, signal, open_channel):
        trans_sig = signal.transpose(self.axes)

        return trans_sig, open_channel

















