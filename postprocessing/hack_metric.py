import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
import pickle

def hack_f1(prob, open_channels):
    """
    prob : (n_data, n_channel)
    """
    pred = np.argmax(prob, axis=1) # (n_data)

    #calib_prob = calc_calib_prob(prob, open_channels)
    temperature = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3]
    print(temperature)
    calib_prob = change_prob_temp(prob, temperature)
    #calib_prob = prob
    
    pred = _hack_f1(calib_prob, open_channels, pred)
    print()
    pred = _hack_f1(calib_prob, open_channels, pred)
    print()
    pred = _hack_f1(calib_prob, open_channels, pred)
    print()
    pred = _hack_f1(calib_prob, open_channels, pred)
    print()
    pred = _hack_f1(calib_prob, open_channels, pred)

    return pred

def _hack_f1(prob, open_channels, pred):
    idxs_for_debug = np.linspace(0, 5000000-1, 250).astype('int')

    _, tp_fp = np.unique(pred, return_counts=True) # (n_channel)
    
    tp = calc_sum_prob(prob, pred)
    tp_fn = calc_sum_prob(prob)

    print(tp / tp_fp)

    f1 = 2 * tp / (tp_fp + tp_fn)
    f1_add_self = 2 * (tp + 1) / (tp_fp + 1 + tp_fn + 1)
    f1_falt = 2 * tp / (tp_fp + 1 + tp_fn)

    inc_macro_f1 = f1_add_self - f1
    dec_macro_f1 = f1_falt - f1

    expect_inc_macro_f1 = inc_macro_f1 * prob
    expect_dec_macro_f1 = dec_macro_f1 * prob
    expect_dec_macro_f1 = np.sum(expect_dec_macro_f1, axis=1, keepdims=True) - expect_dec_macro_f1

    expect_delta_macro_f1 = expect_inc_macro_f1 + expect_dec_macro_f1

    a = expect_delta_macro_f1[idxs_for_debug]

    mod_pred = np.argmax(expect_delta_macro_f1, axis=1)
    print(tp_fp)
    print(np.unique(mod_pred, return_counts=True)[1])

    return mod_pred

def calc_sum_prob(prob, pred=None):
    if pred is not None:
        sum_prob = np.eye(11)[pred]
        sum_prob = sum_prob * prob
    else:
        sum_prob = prob

    sum_prob = np.sum(sum_prob, axis=0) # (n_channel)
    return sum_prob

def change_prob_temp(prob, temp):
    cng_prob = np.power(prob, temp)
    cng_prob = cng_prob / np.sum(cng_prob, axis=1, keepdims=True)
    return cng_prob

def calc_calib_prob(prob, open_channels):
    #check_estimator(IdentifyEstimator)
    base_est = IdentifyEstimator()

    calibrated_clf = CalibratedClassifierCV(base_est, method='isotonic', cv='prefit')
    calibrated_clf.fit(prob, open_channels)
    print(calibrated_clf.score(prob, open_channels))

    pickle.dump(calibrated_clf, open('calibrated_clf', 'wb'))

    clb_p = calibrated_clf.predict_proba(prob)
    return clb_p

class IdentifyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=11):
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)
        return

    def fit(self, x, y):
        return self 

    def predict(self, x):
        return np.argmax(x, axis=-1)

    def predict_proba(self, x):
        return x

    def score(self, x, y):
        return 1

    def get_params(self, deep=True):
        return {'n_classes': self.n_classes}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

def hack_f1_v1(prob):
    """
    prob : (n_data, n_channel)
    """
    idxs_for_debug = np.linspace(0, 5000000-1, 250).astype('int')

    pred = np.argmax(prob, axis=1) # (n_data)
    _, predN = np.unique(pred, return_counts=True) # (n_channel)

    #temperature = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    temperature = 8
    print(temperature)
    calib_prob = change_prob_temp(prob, temperature)
    
    sum_prob = np.eye(11)[pred]
    sum_prob = sum_prob * calib_prob
    sum_prob = np.sum(sum_prob, axis=0) # (n_channel)
    print(sum_prob / predN)

    f1 = sum_prob / predN
    f1_add_self = (sum_prob + 1) / (predN + 1)
    f1_falt = sum_prob / (predN + 1/2)

    inc_macro_f1 = f1_add_self - f1
    dec_macro_f1 = f1_falt - f1

    expect_inc_macro_f1 = inc_macro_f1 * calib_prob
    expect_dec_macro_f1 = dec_macro_f1 * calib_prob
    expect_dec_macro_f1 = np.sum(expect_dec_macro_f1, axis=1, keepdims=True) - expect_dec_macro_f1

    expect_delta_macro_f1 = expect_inc_macro_f1 + expect_dec_macro_f1

    a = expect_delta_macro_f1[idxs_for_debug]

    mod_pred = np.argmax(expect_delta_macro_f1, axis=1)
    print(predN)
    print(np.unique(mod_pred, return_counts=True)[1])

    return mod_pred

def hack_f1_v2(prob, open_channels):
    """
    prob : (n_data, n_channel)
    """
    idxs_for_debug = np.linspace(0, 5000000-1, 250).astype('int')

    pred = np.argmax(prob, axis=1) # (n_data)
    _, predN = np.unique(pred, return_counts=True) # (n_channel)

    calib_prob = calc_calib_prob(prob, open_channels)
    
    sum_prob = np.eye(11)[pred]
    sum_prob = sum_prob * calib_prob
    sum_prob = np.sum(sum_prob, axis=0) # (n_channel)
    print(sum_prob / predN)

    f1 = sum_prob / predN
    f1_add_self = (sum_prob + 1) / (predN + 1)
    f1_falt = sum_prob / (predN + 1/2)

    inc_macro_f1 = f1_add_self - f1
    dec_macro_f1 = f1_falt - f1

    expect_inc_macro_f1 = inc_macro_f1 * calib_prob
    expect_dec_macro_f1 = dec_macro_f1 * calib_prob
    expect_dec_macro_f1 = np.sum(expect_dec_macro_f1, axis=1, keepdims=True) - expect_dec_macro_f1

    expect_delta_macro_f1 = expect_inc_macro_f1 + expect_dec_macro_f1

    a = expect_delta_macro_f1[idxs_for_debug]

    mod_pred = np.argmax(expect_delta_macro_f1, axis=1)
    print(predN)
    print(np.unique(mod_pred, return_counts=True)[1])

    #mod_pred = np.argmax(calib_prob, axis=1)

    return mod_pred
