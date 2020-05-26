import pandas as pd
import numpy as np

def make_submission(time, preds, filename_head=''):
    print('time len, preds len : {0}, {1}'.format(len(time), len(preds)))

    sub_df = pd.concat([pd.Series(time), pd.Series(preds)], axis=1)
    sub_df.columns = ['time', 'open_channels']
    sub_df['time'] = sub_df['time'].map(lambda x: '%.4f' % x)

    sub_df.to_csv(filename_head + 'submission.csv', index=False)

    print(sub_df.head())
    return

def make_prediction_result(time, signal, open_channels, preds, filename_head=''):
    if open_channels is None:
        open_channels = np.ones_like(preds) * (-1)

    print('len times, signal, open_channels, preds : {0}, {1}, {2}, {3}'.format(len(time), len(signal), len(open_channels), len(preds)))

    sub_df = pd.concat([pd.Series(time), pd.Series(signal), pd.Series(open_channels), pd.Series(preds)], axis=1)
    sub_df.columns = ['time', 'signal', 'open_channels', 'preds']
    sub_df['time'] = sub_df['time'].map(lambda x: '%.4f' % x)

    sub_df.to_csv(filename_head + 'pred_result.csv', index=False)

    print(sub_df.head())
    return