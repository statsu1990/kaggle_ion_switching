import numpy as np
from sklearn.metrics import f1_score

def my_macro_f1_score(y_true, y_pred, n_labels=11):
    total_f1 = 0.
    f1s = []
    for i in range(n_labels):
        yt = y_true == i
        yp = y_pred == i

        tp = np.sum(yt & yp)

        tpfp = np.sum(yp)
        tpfn = np.sum(yt)
        if tpfp == 0:
            print('[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples.')
            precision = 0.
        else:
            precision = tp / tpfp
        if tpfn == 0:
            print(f'[ERROR] label not found in y_true...')
            recall = 0.
        else:
            recall = tp / tpfn

        if precision == 0. or recall == 0.:
            f1 = 0.
        else:
            f1 = 2 * precision * recall / (precision + recall)
        total_f1 += f1
        f1s.append(f1)

    total_f1 = total_f1 / n_labels

    return total_f1, f1s

def print_summary(macro_f1, f1s):
    print('macro_f1 : {:}'.format(macro_f1))
    for i, f1 in enumerate(f1s):
        print('f1 label {:} : {:.4f}'.format(i, f1))
    return

def _calc_macro_f1(y_true, y_pred, print_log=False):
    #macro_f1 = f1_score(y_true, y_pred, average='macro')
    #if print_log:
    #    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    #    print_summary(macro_f1, f1s)
    macro_f1, f1s = my_macro_f1_score(y_true, y_pred, n_labels=11)
    if print_log:
        print_summary(macro_f1, f1s)

    return macro_f1

def macro_f1(y_true, y_pred, group=None, print_log=False):
    if group is None:
        macro_f1 = _calc_macro_f1(y_true, y_pred, print_log)
        return macro_f1
    else:
        macro_f1s = []
        uniq_grs = np.unique(group)
        for gr in uniq_grs:
            if print_log:
                print('------\ngroup {0}'.format(gr))
            tg_idxs = np.arange(len(group))[group == gr]
            macro_f1 = _calc_macro_f1(y_true[tg_idxs], y_pred[tg_idxs], print_log)
            macro_f1s.append(macro_f1)
        return macro_f1s


