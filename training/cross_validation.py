import pandas as pd
import numpy as np

def kfold_2split(group, split_rate=0.8):
    """
    Args:
        group : time series of group
        split_rate : before half rate
    Returns:
        before_half_idxs : split_rate
        after_half_idxs : 1 - split_rate
    """

    before_half_idxs = []
    after_half_idxs = []

    num_group = len(np.unique(group))
    start_idx = 0
    for ig in range(num_group):
        num_elem = np.sum(group == ig)
        num_before_half = int(num_elem * split_rate)

        before_half_idxs.append(np.arange(start_idx, start_idx + num_before_half))
        after_half_idxs.append(np.arange(start_idx + num_before_half, start_idx + num_elem))

        start_idx += num_elem

    before_half_idxs = np.concatenate(before_half_idxs)
    after_half_idxs = np.concatenate(after_half_idxs)

    return before_half_idxs, after_half_idxs

