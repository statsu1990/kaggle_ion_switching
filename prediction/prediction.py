import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

def real_to_integer(value, min=0, max=10):
    return np.clip(np.round(value).astype('int'), min, max)

def predict(net, loader, classification=True, soft_value=False):
    net.eval()

    if soft_value:
        softmax = nn.Softmax(dim=-1)

    preds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):

            if len(data) == 2 or len(data) == 3:
                signals, group = data[0].cuda(), data[1].cuda()
            else:
                signals, group, open_channels, trams_mtrxs = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            if len(data) == 2 or len(data) == 3:
                outputs = net(signals)
            else:
                outputs = net(signals, trams_mtrxs)

            if classification:
                if soft_value:
                    preds.append(softmax(outputs).cpu().numpy().reshape(-1,outputs.size()[-1]))
                else:
                    preds.append(np.ravel(outputs.max(2)[1].cpu().numpy()))
            else:
                # regression
                preds.append(np.ravel(real_to_integer(outputs.cpu().numpy())))

    preds = np.concatenate(preds)
    print('predict point num : {0}'.format(len(preds)))

    return preds

def predict_v2(net, loader, classification=True, soft_value=False):
    net.eval()

    if soft_value:
        softmax = nn.Softmax(dim=-1)

    preds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):

            if len(data) == 2 or len(data) == 3:
                signals, group = data[0].cuda(), data[1].cuda()
            else:
                signals, group, open_channels, trams_mtrxs = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            if len(data) == 2 or len(data) == 3:
                outputs = net(signals)
            else:
                outputs = net(signals, trams_mtrxs)

            sig_arg = np.argsort(signals.cpu().numpy(), axis=1)
            pred = outputs.max(2)[1].cpu().numpy()
            pred_label = sig_arg[:,-1,:]
            pred_label[pred==0] = (sig_arg[:,-2,:])[pred==0]
            preds.append(np.ravel(pred_label))

    preds = np.concatenate(preds)
    print('predict point num : {0}'.format(len(preds)))

    return preds