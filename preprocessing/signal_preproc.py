import pandas as pd
import numpy as np
import scipy as sp
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy import signal as spsig
from scipy import ndimage
from tqdm import tqdm
import math

def conv_filter(signal, window_size, filter='gaussian', std=None, num_filtering=1):
    """
    Args:
        filter : 'gaussian', 'average'
    """
    if filter == 'gaussian':
        std = std if std is not None else (window_size - 1) / 4
        w = spsig.gaussian(window_size, std)
        w = w / np.sum(w)
    elif filter == 'average':
        w = np.ones(window_size) / window_size

    filtered_sig = signal.copy()
    for i in range(num_filtering):
        filtered_sig = np.pad(filtered_sig, (window_size//2, window_size//2), 'reflect')
        filtered_sig = np.convolve(filtered_sig, w, 'valid')

    #print('size signal / filtered signal : {0} / {1}'.format(len(signal), len(filtered_sig)))

    return filtered_sig

def gaussian_filter(signal, std, num_filtering=1):
    filtered_sig = signal.copy()
    for i in range(num_filtering):
        filtered_sig = ndimage.gaussian_filter(filtered_sig, std, mode='reflect')
    return filtered_sig

def to_const_filter(signal, filter='median'):
    if filter == 'median':
        const = np.median(signal)
    elif filter == 'average':
        const = np.average(signal)
    filtered_sig = np.ones_like(signal) * const
    return filtered_sig

def open_channel_filter(signal, open_channels, oc_to_use=None):
    if oc_to_use is None:
        uni_oc, count = np.unique(open_channels, return_counts=True)
        oc_to_use = uni_oc[np.argmax(count)]

    filtered_sig = signal.copy()
    filtered_sig[open_channels != oc_to_use] = np.nan

    filtered_sig = pd.Series(filtered_sig)
    filtered_sig = filtered_sig.interpolate(method='linear', limit_direction='both')
    filtered_sig = filtered_sig.interpolate(method='linear', limit_direction='forward')
    filtered_sig = filtered_sig.interpolate(method='linear', limit_direction='backward')
    filtered_sig = filtered_sig.values

    return filtered_sig

def shift(signal, n):
    fill_val = signal[0] if n > 0 else signal[-1]
    shifted_sig = np.ones_like(signal) * fill_val

    if n > 0:
        shifted_sig[n:] = signal[:-n]
    else:
        shifted_sig[:n] = signal[-n:]

    return shifted_sig

def max_log_likelihood(init_value, signal, serch_range, n_div, trunc_range):
    """
    https://www.kaggle.com/statsu/average-signal
    calculate maximum log likelihood near init_value.
    """
    xgrid = np.linspace(init_value-serch_range, init_value+serch_range, n_div)
    logll_max = None
    x_max = None
    for x in xgrid:
        tg_sig = signal[np.abs(signal - x) < trunc_range]
        logll = - np.average((tg_sig - x)**2) / 2
        
        if logll_max is None:
            logll_max = logll
            x_max = x
        elif logll_max < logll:
            logll_max = logll
            x_max = x

    return x_max

def distance_from_ave_wo_label(signal, serch_range, n_div, trunc_range, max_channel, sig_dist, dist_coef):
    init_value = np.median(signal)
    base_ave_sig = max_log_likelihood(init_value, signal, serch_range, n_div, trunc_range)
    print('base_ave_sig ', base_ave_sig)

    # average signals of each open channels
    ave_sigs = base_ave_sig + np.arange(-max_channel, max_channel + 1) * sig_dist

    # signal : (time,)
    # ave_sigs : (max_channel*2-1,)
    # distance of average signals of each open channels
    dists = np.exp(- (signal[:,None] - ave_sigs[None,:])**2 / sig_dist**2 * dist_coef) # (time, max_channel*2-1)

    return dists

def distance_from_ave_with_label(signal, open_channels, max_channel, sig_dist, dist_coef, use_ave=True, use_middle=False):
    uni_oc, count = np.unique(open_channels, return_counts=True)

    # calc base channel and average signal
    base_oc = uni_oc[np.argmax(count)]
    if use_ave:
        base_ave_sig = np.average(signal[open_channels==base_oc])
    else:
        base_ave_sig = np.median(signal[open_channels==base_oc])

    # calc distance of average signals of each open channels
    if sig_dist is None:
        second_oc = uni_oc[np.argsort(count)[-2]]
        if use_ave:
            second_ave_sig = np.average(signal[open_channels==second_oc])
        else:
            second_ave_sig = np.median(signal[open_channels==second_oc])
        sig_dist = np.abs(base_ave_sig - second_ave_sig) / np.abs(base_oc - second_oc)
     
    ave_sigs = np.arange(0, max_channel+1) * sig_dist - base_oc * sig_dist + base_ave_sig

    # middle
    if use_middle:
        asigs = []
        for i in range(len(ave_sigs)):
            asigs.append(ave_sigs[i])
            if i < len(ave_sigs) - 1:
                asigs.append((ave_sigs[i] + ave_sigs[i+1])*0.5)
        ave_sigs = np.array(asigs)

    # calc dist_coef
    if dist_coef is None:
        tg_sig = signal[open_channels==base_oc]
        if use_ave:
            s = np.std(tg_sig)
        else:
            # normalized interquartile range
            s = (np.percentile(tg_sig, 75) - np.percentile(tg_sig, 25)) * 0.5 * 1.3490
        dist_coef = 1.0 / (2.0 * s ** 2) * sig_dist**2

    # signal : (time,)
    # ave_sigs : (max_channel*2-1,)
    # distance of average signals of each open channels
    dists = np.exp(- (signal[:,None] - ave_sigs[None,:])**2 / sig_dist**2 * dist_coef) # (time, max_channel*2-1)

    return dists

def apply_each_group(signal, group, func, args, open_channels=None):
    num_groups = len(np.unique(group))

    sigs = []
    start_idx = 0
    for gr in tqdm(range(num_groups)):
        num_element = np.sum(group == gr)

        if open_channels is None:
            sig = signal[start_idx : start_idx+num_element]
            sig = func(sig, *args)
        else:
            sig = signal[start_idx : start_idx+num_element]
            oc = open_channels[start_idx : start_idx+num_element]
            sig = func(sig, oc, *args)
        sigs.append(sig)

        start_idx += num_element
    sigs = np.concatenate(sigs)

    return sigs

def plot_signal(signal):
    res = 1
    plt.figure(figsize=(20,5))
    plt.plot(range(0,len(signal), res), signal[0::res])
    plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16)
    plt.show()
    return

class PreProcess_v1:
    def __init__(self):
        self.signal_average = np.array([1.386246,]).astype('float32')
        self.signal_std = np.array([3.336219,]).astype('float32')
        self.input_channels = 1

    def preprocessing(self, data_df):
        # signal
        sig = data_df.signal.values.astype('float32')
        sig = sig[:, None] # (time, channel)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sig, group, open_channels)

        return sig, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v2:
    """
    no implementation
    """
    def __init__(self):
        return

class PreProcess_v3_0_1:
    def __init__(self):
        self.signal_average = np.array([1.3673096e-06,]).astype('float32')
        self.signal_std = np.array([1.139225,]).astype('float32')
        self.input_channels = 1

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        return

    def preprocessing(self, data_df):
        # signal
        sig = data_df.signal.values
        sig = sig - apply_each_group(sig, data_df.group.values, conv_filter, 
                                     [self.window_size, self.filter, self.std, self.num_filtering])
        sig = sig[:, None].astype('float32') # (time, channel)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sig, group, open_channels)
        #plot_signal(sig)

        return sig, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v3_1_5:
    def __init__(self):
        self.signal_average = np.array([1.3700463e-06, 1.3901746e+00]).astype('float32')
        self.signal_std = np.array([1.1374537, 3.1242452]).astype('float32')
        self.input_channels = 2

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        return

    def preprocessing(self, data_df):
        # signal
        sig = data_df.signal.values
        sig2 = apply_each_group(sig, data_df.group.values, conv_filter, 
                                [self.window_size, self.filter, self.std, self.num_filtering])
        sig = np.concatenate([(sig-sig2)[:, None], sig2[:, None]], axis=1).astype('float32') # (time, channel)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sig, group, open_channels)
        #plot_signal(sig)

        return sig, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

# combine before after
class PreProcess_v4_0_1:
    def __init__(self):
        self.signal_average = np.array([1.3901746e+00] + [1.3700463e-06]*11).astype('float32')
        self.signal_std = np.array([3.1242452] + [1.1374537]*11).astype('float32')
        self.input_channels = 12

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        # shift
        self.shift_lens = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)

        return

    def preprocessing(self, data_df):
        # signal
        sigs = []
        sig = data_df.signal.values

        sig2 = apply_each_group(sig, data_df.group.values, conv_filter, 
                                [self.window_size, self.filter, self.std, self.num_filtering])
        sigs.append(sig2[:,None])

        sig = sig - sig2
        sigs.append(sig[:,None])
        for sh in self.shift_lens:
            sigs.append(shift(sig, sh)[:,None])
        sigs = np.concatenate(sigs, axis=1).astype('float32') # (time, channel)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

# use signal center of each channel without label
class PreProcess_v5_0_0:
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 21

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.serch_range = 0.8
        self.n_div = 500
        self.trunc_range = 0.3
        self.max_channel = 10
        self.sig_dist = 1.21
        self.dist_coef = 1.0

        return

    def preprocessing(self, data_df):
        # signal
        sigs = data_df.signal.values
        sigs = sigs - apply_each_group(sigs, data_df.group.values, conv_filter, 
                                [self.window_size, self.filter, self.std, self.num_filtering])

        sigs = apply_each_group(sigs, data_df.group.values, distance_from_ave_wo_label, 
                                     [self.serch_range, self.n_div, self.trunc_range, self.max_channel, self.sig_dist, self.dist_coef])
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

# use signal center of each channel with label
class PreProcess_v6_0_0:
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = 1.21
        self.dist_coef = 4 * math.log(2)
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        sigs = data_df.signal.values
        sigs = sigs - apply_each_group(sigs, data_df.group.values, conv_filter, 
                                [self.window_size, self.filter, self.std, self.num_filtering])

        sigs = apply_each_group(sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v6_0_1:
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = 1.21
        self.dist_coef = math.log(5)
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        sigs = data_df.signal.values
        sigs = sigs - apply_each_group(sigs, data_df.group.values, conv_filter, 
                                [self.window_size, self.filter, self.std, self.num_filtering])

        sigs = apply_each_group(sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v6_0_2:
    def __init__(self):
        self.signal_average = np.array([-0.5]).astype('float32')
        self.signal_std = np.array([0.5]).astype('float32')
        self.input_channels = 11

        # filter
        self.window_size = 10001
        self.filter='gaussian'
        self.std = (self.window_size - 1) / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = 1.21
        self.dist_coef = math.log(5)
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        sigs = data_df.signal.values
        sigs = sigs - apply_each_group(sigs, data_df.group.values, conv_filter, 
                                [self.window_size, self.filter, self.std, self.num_filtering])

        sigs = apply_each_group(sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

# use (gaussian_filter + open_channel_filter), signal center of each channel with label
class PreProcess_v7_0_0:
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = 1.23
        self.dist_coef = math.log(5)
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v7_0_1:
    """
    self.use_ave = False
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = 1.23
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v7_0_2:
    """
    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v7_0_2_test:
    """
    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 100 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v7_0_3:
    """
    self.use_ave = False
    self.sig_dist = None
    self.dist_coef = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = None
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v7_1_0:
    """
    ave_signal_base_open_channel(self.dist_coef = math.log(5))
    ave_signal_base_open_channel(self.dist_coef = None)
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 22

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel1 = 10
        self.sig_dist1 = None
        self.dist_coef1 = math.log(5)
        self.use_ave1 = False

        # ave_signal_base_open_channel
        self.max_channel2 = 10
        self.sig_dist2 = None
        self.dist_coef2 = None
        self.use_ave2 = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        sigs = []
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])
        # ave_signal_base_open_channel(self.dist_coef = math.log(5))
        sigs.append(apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel1, self.sig_dist1, self.dist_coef1, self.use_ave1], ref_open_channels))
        # ave_signal_base_open_channel(self.dist_coef = None)
        sigs.append(apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel2, self.sig_dist2, self.dist_coef2, self.use_ave2], ref_open_channels))
        
        sigs = np.concatenate(sigs, axis=1).astype('float32') # (time, channel*2)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v7_1_1:
    """
    2D
    ave_signal_base_open_channel(self.dist_coef = math.log(5))
    ave_signal_base_open_channel(self.dist_coef = None)
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 2

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel1 = 10
        self.sig_dist1 = None
        self.dist_coef1 = math.log(5)
        self.use_ave1 = False

        # ave_signal_base_open_channel
        self.max_channel2 = 10
        self.sig_dist2 = None
        self.dist_coef2 = None
        self.use_ave2 = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        sigs = []
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])
        # ave_signal_base_open_channel(self.dist_coef = math.log(5))
        sigs.append(apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel1, self.sig_dist1, self.dist_coef1, self.use_ave1], ref_open_channels)[:,:,None])
        # ave_signal_base_open_channel(self.dist_coef = None)
        sigs.append(apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel2, self.sig_dist2, self.dist_coef2, self.use_ave2], ref_open_channels)[:,:,None])
        
        sigs = np.concatenate(sigs, axis=2).astype('float32') # (time, channel, 2)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

# fork PreProcess_v7_0_2
class PreProcess_v8_0_0:
    """
    same PreProcess_v7_0_2

    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 10000 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v8_0_0_test:
    """
    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        self.std = 100 / 4
        self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
                                    [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v8_0_1:
    """
    no gaussian_filter

    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        # open_channel_filter
        self.oc_to_use = None
        # gaussian_filter
        #self.std = 10000 / 4
        #self.num_filtering = 1

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        #fil_sigs = apply_each_group(fil_sigs, data_df.group.values, gaussian_filter, 
        #                            [self.std, self.num_filtering])

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)
        #plot_signal(sig)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

# clean data
class PreProcess_v9_0_0:
    """
    use clean value
    """
    def __init__(self):
        self.signal_average = np.array([0.08261159,]).astype('float32')
        self.signal_std = np.array([2.4877818,]).astype('float32')
        self.input_channels = 1

    def preprocessing(self, data_df):
        # signal
        sig = data_df.signal.values.astype('float32')
        sig = sig[:, None] # (time, channel)

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sig, group, open_channels)

        #plot_signal(open_channels)
        #plot_signal(sig)

        return sig, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v9_1_0:
    """
    clean data
    to_const_filter

    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 11

        self.use_open_channel = True

        # open_channel_filter
        self.oc_to_use = None
        # to_const_filter
        self.filter='median'

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(5)
        self.use_ave = False
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        #plot_signal(fil_sigs)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, to_const_filter, 
                                    [self.filter])
        #plot_signal(fil_sigs)

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))

class PreProcess_v9_2_0:
    """
    clean data
    to_const_filter

    self.use_middle = True
    self.use_ave = False
    self.sig_dist = None
    """
    def __init__(self):
        self.signal_average = np.array([0]).astype('float32')
        self.signal_std = np.array([1]).astype('float32')
        self.input_channels = 21

        self.use_open_channel = True

        # open_channel_filter
        self.oc_to_use = None
        # to_const_filter
        self.filter='median'

        # ave_signal_base_open_channel
        self.max_channel = 10
        self.sig_dist = None
        self.dist_coef = math.log(1000)
        self.use_ave = False
        self.use_middle = True
        return

    def preprocessing(self, data_df, ref_open_channels):
        # signal
        fil_sigs = apply_each_group(data_df.signal.values, data_df.group.values, open_channel_filter, 
                                    [self.oc_to_use], ref_open_channels)
        #plot_signal(fil_sigs)
        fil_sigs = apply_each_group(fil_sigs, data_df.group.values, to_const_filter, 
                                    [self.filter])
        #plot_signal(fil_sigs)

        sigs = apply_each_group(data_df.signal.values - fil_sigs, data_df.group.values, distance_from_ave_with_label, 
                                     [self.max_channel, self.sig_dist, self.dist_coef, self.use_ave, self.use_middle], ref_open_channels)
        sigs = sigs.astype('float32')

        # group
        group = data_df.group.values.astype('int64')

        # open_channels
        if 'open_channels' in data_df.columns:
            open_channels = data_df.open_channels.values.astype('int64')
        elif ref_open_channels is not None:
            open_channels = ref_open_channels.astype('int64')
        else:
            open_channels = None

        # check
        self.check_value(sigs, group, open_channels)

        return sigs, group, open_channels

    def check_value(self, sig, group, open_channels):
        print('ave : data {0} / constant {1}'.format(np.average(sig, axis=0), self.signal_average))
        print('std : data {0} / constant {1}'.format(np.std(sig, axis=0), self.signal_std))


