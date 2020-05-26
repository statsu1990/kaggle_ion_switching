import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _get_soft_target(label, num_class=100, label_smooth=0.0):
    soft_tg = (torch.eye(num_class)[label] * (1- label_smooth) + label_smooth / num_class).cuda()
    return soft_tg

def _get_mixed_soft_target(label_a, label_b, mix_rate, num_class=100, label_smooth=0.0):
    soft_tg_a = (torch.eye(num_class)[label_a] * (1- label_smooth) + label_smooth / num_class).cuda()
    soft_tg_b = (torch.eye(num_class)[label_b] * (1- label_smooth) + label_smooth / num_class).cuda()
    return mix_rate * soft_tg_a + (1 - mix_rate) * soft_tg_b

def _CrossEntropyLossWithSoftTarget(input, soft_target, weight):
    """
    Args:
        input: shape (num batch, num class)
        soft_target: shape (num batch, num class)
        weight: shape (num batch)
    """

    loss = - torch.mean(weight * torch.sum(F.log_softmax(input, dim=-1) * soft_target, dim=-1))
    return loss

def _MSE(input, target, weight):
    loss = torch.mean(weight * (input.view(target.size()[0], -1) - target) ** 2) / 2
    return loss

class MyCrossEntropyLoss(nn.Module):
    def __init__(self, num_class, label_smooth=0.0):
        super(MyCrossEntropyLoss, self).__init__()

        self.num_class = num_class
        self.label_smooth = label_smooth

    def forward(self, input, label, label_b=None, mix_rate=None):
        if (label_b is None) or (mix_rate is None):
            soft_target = _get_soft_target(label, self.num_class, self.label_smooth)
        else:
            soft_target = _get_mixed_soft_target(label, label_b, mix_rate, self.num_class, self.label_smooth)

        loss = _CrossEntropyLossWithSoftTarget(input, soft_target, weight=1.0)
        return loss

class F1Loss(nn.Module):
    """
    https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    """
    def __init__(self, num_class, label_smooth=0.0):
        super(F1Loss, self).__init__()

        self.num_class = num_class
        self.label_smooth = label_smooth
        self.eps = 1e-3

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logit, label):
        reduce_dim = list(range(len(logit.size()) - 1))

        soft_target = _get_soft_target(label, self.num_class, self.label_smooth)
        pred = self.softmax(logit)

        tp = torch.mean(soft_target * pred, dim=reduce_dim)
        #tn = torch.mean((1-soft_target) * (1-pred), dim=reduce_dim)
        fp = torch.mean((1-soft_target) * pred, dim=reduce_dim)
        fn = torch.mean(soft_target * (1-pred), dim=reduce_dim)

        p = tp / (tp + fp + self.eps)
        r = tp / (tp + fn + self.eps)

        f1 = 2 * p * r / (p + r + self.eps)
        f1 = torch.mean(f1)
        return 1 - f1

class ClassBalanced_CrossEntropyLoss(nn.Module):
    """
    https://arxiv.org/abs/1901.05555
    """
    def __init__(self, reference_labels, num_class, beta=0.999, label_smooth=0.0):
        super(ClassBalanced_CrossEntropyLoss, self).__init__()

        self.num_class = num_class
        self.beta = beta
        self.label_smooth = label_smooth

        # shape (1, num class)
        self.weight_per_cls = self.__calc_weight_per_cls(reference_labels)
        self.weight_per_cls = self.weight_per_cls[np.newaxis, :]
        self.weight_per_cls = torch.from_numpy(self.weight_per_cls).cuda()

        return

    def __count_per_class(self, labels, num_class):
        unique_labels, count = np.unique(labels, return_counts=True)

        c_per_cls = np.zeros(num_class)
        c_per_cls[unique_labels] = count

        return c_per_cls

    def __calc_weight_per_cls(self, labels):
        eps = 1e-2
        eps2 = 1e-6

        # effective number
        c_per_cls = self.__count_per_class(labels, self.num_class)
        ef_Ns = (1 - np.power(self.beta, c_per_cls + eps)) / (1 - self.beta)

        # weight
        weight = 1 / (ef_Ns + eps2)

        # normalize weight
        WN = np.sum(weight * c_per_cls)
        N = np.sum(c_per_cls)
        weight = weight * N / WN

        return weight

    def __call__(self, output, label, label_b=None, mix_rate=None):
        if (label_b is None) or (mix_rate is None):
            soft_tg = _get_soft_target(label, self.num_class, self.label_smooth)
        else:
            soft_tg = _get_mixed_soft_target(label, label_b, mix_rate, self.num_class, self.label_smooth)

        weight = torch.sum(soft_tg * self.weight_per_cls, dim=-1)

        loss = _CrossEntropyLossWithSoftTarget(output, soft_tg, weight)

        return loss

class ClassBalanced_MSE(nn.Module):
    """
    https://arxiv.org/abs/1901.05555
    """
    def __init__(self, reference_labels, num_class, beta=0.999):
        super(ClassBalanced_MSE, self).__init__()

        self.num_class = num_class
        self.beta = beta
        self.label_smooth = 0.0

        # shape (1, num class)
        self.weight_per_cls = self.__calc_weight_per_cls(reference_labels)
        self.weight_per_cls = self.weight_per_cls[np.newaxis, :]
        self.weight_per_cls = torch.from_numpy(self.weight_per_cls).cuda()

        return

    def __count_per_class(self, labels, num_class):
        unique_labels, count = np.unique(labels, return_counts=True)

        c_per_cls = np.zeros(num_class)
        c_per_cls[unique_labels] = count

        return c_per_cls

    def __calc_weight_per_cls(self, labels):
        eps = 1e-2
        eps2 = 1e-6

        # effective number
        c_per_cls = self.__count_per_class(labels, self.num_class)
        ef_Ns = (1 - np.power(self.beta, c_per_cls + eps)) / (1 - self.beta)

        # weight
        weight = 1 / (ef_Ns + eps2)

        # normalize weight
        WN = np.sum(weight * c_per_cls)
        N = np.sum(c_per_cls)
        weight = weight * N / WN

        return weight

    def __call__(self, output, label, label_b=None, mix_rate=None):
        if (label_b is None) or (mix_rate is None):
            soft_tg = _get_soft_target(label, self.num_class, self.label_smooth)
            target = label
        else:
            soft_tg = _get_mixed_soft_target(label, label_b, mix_rate, self.num_class, self.label_smooth)
            target = label * mix_rate + label_b * (1 - mix_rate)

        weight = torch.sum(soft_tg * self.weight_per_cls, dim=-1)

        loss = _MSE(output, target, weight)

        return loss