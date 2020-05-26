import os
import numpy as np
import pandas as pd


data_dir = '../input/liverpool-ion-switching'

def get_original_data(is_train=True, cleaned=False):
    """
    Returns:
        columns=[time, signal, open_channels]
    """
    if is_train:
        if cleaned:
            file = 'train_clean.csv'
        else:
            file = 'train.csv'
    else:
        if cleaned:
            file = 'test_clean.csv'
        else:
            file = 'test.csv'
    file = os.path.join(data_dir, file)

    df = pd.read_csv(file)

    print('data length : {0}'.format(len(df)))

    return df

def get_10open_channel_group(train_original_df):
    df1 = train_original_df[2000000:2500000]
    df2 = train_original_df[4500000:5000000]
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    return df

def add_group(df, unit_length):
    """
    Returns:
        columns=[time, signal, open_channels, group]
    """
    group = np.arange(len(df))
    group = group // unit_length

    print('num group : {0}'.format(len(np.unique(group))))

    group = pd.DataFrame(group[:,None], columns=['group'])
    added_df = pd.concat([df, group], axis=1)
    return added_df

def _split_into_section(values, group, length, stride):
    """
    Args:
        values : (time, channel) or (time)
        group : (time)
    """

    splited_vs = []
    splited_groups = []

    num_group = len(np.unique(group))
    idx = 0
    for ig in range(num_group):
        num_elem = np.sum(group == ig)
        for i in range(num_elem):
            if (i + length) <= num_elem and i % stride == 0:
                splited_vs.append(values[idx:idx+length])
                splited_groups.append(group[idx])
            idx += 1

    splited_vs = np.array(splited_vs)
    splited_groups = np.array(splited_groups).astype('int')

    # check
    print('splited_values shape : {0}'.format(splited_vs.shape))
    sp_num_gr = len(np.unique(splited_groups))
    for ig in range(sp_num_gr):
        print('num data in splited group {0} : {1}'.format(ig, np.sum(splited_groups == ig)))

    return splited_vs, splited_groups

def split_into_section(signal, group, open_channel, length, stride):
    splited_signals, splited_groups = _split_into_section(signal, group, length, stride)

    if open_channel is not None:
        splited_op_chn, splited_groups = _split_into_section(open_channel, group, length, stride)
        return splited_signals, splited_groups, splited_op_chn
    else:
        return splited_signals, splited_groups












