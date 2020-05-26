import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from data import data_utils, metric
from preprocessing import signal_preproc
from model import transformer, loss, signal_encoder as sig_enc
from training import training, dataset, cross_validation, sig_transform
from prediction import prediction, prediction_utils
from postprocessing import hack_metric

trained_models_dir = '../trained_models'

def get_checkpoint(path):
    cp = torch.load(path, map_location=lambda storage, loc: storage)
    return cp

# clean data
class Model_v5_0_0:
    """
    ep 80, val 0.9378
    clean data
    PreProcess_v9_0_0

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.sig_preproc = signal_preproc.PreProcess_v9_0_0()

        self.crop_signal_length = 500
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        kernel_size = 5
        num_block = 3
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=None, mu_gene=None))
        use_label = False

        dropout = 0.1
        positional_encoder = nn.Dropout(dropout)
        #positional_encoder = None

        nhead = 2
        nhid = 64
        dropout = 0.1
        nlayers = 2
        transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        #transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=3, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=False)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=False)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_1:
    """
    ep 80, val 0.938800951

    clean data
    PreProcess_v9_1_0

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 500
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        dropout = 0.1
        positional_encoder = nn.Dropout(dropout)
        #positional_encoder = None

        nhead = 2
        nhid = 64
        dropout = 0.1
        nlayers = 2
        transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        #transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=3, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_2:
    """
    ep 80, 0.9388

    clean data
    PreProcess_v9_1_0

    sig_transform.RandomFlip(p=0.0),

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 500
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        dropout = 0.1
        positional_encoder = nn.Dropout(dropout)
        #positional_encoder = None

        nhead = 2
        nhid = 64
        dropout = 0.1
        nlayers = 2
        transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        #transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.0),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=3, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_3:
    """
    ep 80, val 0.9391

    clean data
    PreProcess_v9_1_0

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 500
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=3, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_4:
    """
    ep 80, val 0.9389

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 100

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=3, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5:
    """
    ep 80, val 0.9394

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_1:
    """
    ep 80, val 0.9388

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    no transmatrix

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = False
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.use_open_channel_for_preproc = self.sig_preproc.use_open_channel
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        #sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
        #                                            common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        #mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
        #                                            common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        sig_gene = None
        mu_gene = None

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_open_channel_for_preproc:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_open_channel_for_preproc:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans, use_trans_matrix=self.use_trans_matrix)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_2:
    """
    ep 80, val 0.9388

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 256
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_3:
    """
    ep 80, val 0.9388

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=4
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=4, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=4, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_4:
    """
    ep 80, val 0.9390

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4
    num_block = 2

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 2
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_5:
    """
    ep 80, val 0.93918767

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_6:
    """
    ep 80, val 0.938959602

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.3), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.3), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_7:
    """
    ep 80, val 0.938799268

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=1.0)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=1.0),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_8:
    """
    ep 80, val 0.938846411

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.3), max_size=1, fill_value=0, p=1.0)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.3), max_size=1, fill_value=0, p=1.0),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_5_9:
    """
    ep 80, val 0.9390

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    no Cutout
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              #sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_6:
    """
    ep 150, val 0.93912788

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_7:
    """
    ep 150, val 0.9393

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_8:
    """
    ep 150, val 0.9396

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_8_1:
    """
    ep 150, val 0.9396

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    self.checkpoint_path = 'Model_v5_0_8_checkpoint'
    self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = 'Model_v5_0_8_checkpoint' #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_9:
    """
    ep 150, val 0.939411477

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 256
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 256

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_10:
    """
    ep 80, val 0.939101253030469

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200
    prob_temperature = 0.5

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11
        self.prob_temperature = 0.5

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix, prob_temperature=self.prob_temperature)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix, prob_temperature=self.prob_temperature)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans, prob_temperature=self.prob_temperature)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_11:
    """
    ep 80, val 0.938978728

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200
    prob_temperature = 0.2

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11
        self.prob_temperature = 0.5

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix, prob_temperature=self.prob_temperature)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix, prob_temperature=self.prob_temperature)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans, prob_temperature=self.prob_temperature)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_0_12:
    """
    ep 150, val 0.9394

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        dropout = 0.1
        positional_encoder = nn.Dropout(dropout)
        #positional_encoder = None

        nhead = 2
        nhid = 128
        dropout = 0.1
        nlayers = 2
        transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        #transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_1_0:
    """
    ep 80, val 0.9393

    clean data
    PreProcess_v9_2_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_2_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_1_1:
    """
    ep 80, val 0.9390

    clean data
    PreProcess_v9_2_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_2_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return


class Model_v5_2_0_0:
    """
    ep 80, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)

    model for small open channels
    sample_weight_per_group=(0.25, 0.25, 0.25, 2.0, 0.25, 1.0, 0.25, 0.0, 1.0, 0.25,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_5_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 2.0 # group 15-19 : ch 0-3
        w_gr20_24 = 0.25 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 0.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 0.25 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_2_0_1:
    """
    ep 80, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)

    model for large open channels
    sample_weight_per_group=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_5_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.1 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.1 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.1 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 0.25 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.1 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 0.25 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return


class Model_v5_3_0:
    """
    ep 80, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 2

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 2.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 0.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])

        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss([0]*1+[1], self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss([0]*1+[1], self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model_v2(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict_v2(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_3_1:
    """
    ep 80, val 0.9372

    binary classify

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 2

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.1 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.1 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.1 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 2.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.1 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 0.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 64

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])

        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss([0]*1+[1], self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss([0]*1+[1], self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model_v2(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict_v2(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v5_3_2:
    """
    ep 80, val 

    binary classify

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 2

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_0_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.1 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.1 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.1 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 2.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.1 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 0.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 3
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 16
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.8
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])

        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss([0]*1+[1], self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss([0]*1+[1], self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 80

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 70, 160], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model_v2(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict_v2(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return


class Model_v6_0_0:
    """
    ep 150, val 0.936270288624983

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v6_0_1:
    """
    ep 150, val 0.936626869

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v6_0_2:
    """
    ep 150, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 500
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 500
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v6_0_3:
    """
    ep 150, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.5 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.5 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.5 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 2.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.5 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 0.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v6_1_0:
    """
    ep 150, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.2 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.2 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.2 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.2 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v6_1_1:
    """
    ep 150, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 250
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 250
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.2 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.2 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.2 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.2 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.2), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

class Model_v6_1_2:
    """
    ep 150, val 

    clean data
    PreProcess_v9_1_0

    self.crop_signal_length = 500
    self.tr_signal_stride = 200

    num_block = 5
    nfeature = 128
    ResNetTransmat1D
    num_convs=2
    channels=nfeature*4

    no TransEncorder

    sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5)
    sample_weight_per_group=(0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0,)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None #self.filename_head + 'checkpoint' #os.path.join(trained_models_dir, self.__class__.__name__, self.filename_head + 'checkpoint')

        self.tr_unit_length = 100000
        self.ts_unit_length = 100000
        self.n_label = 11

        self.use_cleaned_data = True
        self.use_trans_matrix = True
        self.sig_preproc = signal_preproc.PreProcess_v9_1_0()
        self.ref_tr_open_channel = None
        self.ref_ts_open_channel = pd.read_csv('Model_v5_0_8_ts_pred_result.csv').preds.values

        self.crop_signal_length = 500
        self.tr_signal_stride = 200

        #self.sample_weight_per_group = None
        w_gr0_4   = 0.2 # group 0-4   : ch 0-1 (dominant 0)
        w_gr5_9   = 0.2 # group 5-9   : ch 0-1 (dominant 0)
        w_gr10_14 = 0.2 # group 10-14 : ch 0-1 (dominant 1)
        w_gr15_19 = 1.0 # group 15-19 : ch 0-3
        w_gr20_24 = 1.0 # group 20-25 : ch 0-10
        w_gr25_29 = 1.0 # group 25-29 : ch 0-5
        w_gr30_34 = 0.2 # group 30-34 : ch 0-1 (dominant 1)
        w_gr35_39 = 1.0 # group 35-39 : ch 0-3
        w_gr40_44 = 1.0 # group 40-44 : ch 0-5
        w_gr45_49 = 1.0 # group 45-49 : ch 0-10
        self.sample_weight_per_group = np.array([
                                                 w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   w_gr0_4,   # group 0-4   : ch 0-1 (dominant 0)
                                                 w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   w_gr5_9,   # group 5-9   : ch 0-1 (dominant 0)
                                                 w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, w_gr10_14, # group 10-14 : ch 0-1 (dominant 1)
                                                 w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, w_gr15_19, # group 15-19 : ch 0-3
                                                 w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, w_gr20_24, # group 20-25 : ch 0-10
                                                 w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, w_gr25_29, # group 25-29 : ch 0-5
                                                 w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, w_gr30_34, # group 30-34 : ch 0-1 (dominant 1)
                                                 w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, w_gr35_39, # group 35-39 : ch 0-3
                                                 w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, w_gr40_44, # group 40-44 : ch 0-5
                                                 w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, w_gr45_49, # group 45-49 : ch 0-10
                                                 ])

        return

    def get_model(self):
        ## model
        nfeature = 128

        num_block = 5
        sig_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*4, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)
        mu_gene = sig_enc.TransMatrixFilmGenerator(nfeature, num_block, kernel_size=5, channels=nfeature*2, num_convs=2, 
                                                    common_neurons=(nfeature*4,), head_neurons=(nfeature*2,), dropout_rate=0.1)

        kernel_size = 5
        num_block = num_block
        input_channel = self.sig_preproc.input_channels
        gn_group = 32
        se_reduction = 4
        signal_encoder = sig_enc.SignalEncoder(sig_enc.ResNetTransmat1D(nfeature, kernel_size, num_block, input_channel, 
                                                                        gn_group, se_reduction, 
                                                                        sig_gene=sig_gene, mu_gene=mu_gene))
        use_label = False

        #dropout = 0.1
        #positional_encoder = nn.Dropout(dropout)
        positional_encoder = None

        #nhead = 2
        #nhid = 64
        #dropout = 0.1
        #nlayers = 2
        #transformer_encoder = transformer.TransEncorder(nfeature, nhead, nhid, dropout, nlayers)
        transformer_encoder = None

        decoder = transformer.LinearDecoder(nfeature, self.n_label)
        #decoder = None

        model = transformer.SignalTransformer(signal_encoder, positional_encoder, transformer_encoder, decoder)
        
        return model, use_label

    def train_model(self, run_test=True):
        """
        sgd
        """
        FINE_TURNING = False
        CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None

        ## data
        tr_data = data_utils.add_group(data_utils.get_original_data(is_train=True, cleaned=self.use_cleaned_data), self.tr_unit_length)
        if self.use_trans_matrix:
            if self.ref_tr_open_channel is None:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, tr_data.open_channels.values)
            else:
                tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data, self.ref_tr_open_channel)
        else:
            tr_sig, tr_gr, tr_op_chn = self.sig_preproc.preprocessing(tr_data)

        tr_rate = 0.99
        tr_idxs, vl_idxs = cross_validation.kfold_2split(tr_gr, tr_rate)
        vl_sig, vl_gr, vl_op_chn = tr_sig[vl_idxs], tr_gr[vl_idxs], tr_op_chn[vl_idxs]
        tr_sig, tr_gr, tr_op_chn = tr_sig[tr_idxs], tr_gr[tr_idxs], tr_op_chn[tr_idxs]
    
        # split into section
        tr_signal_crop_length = self.crop_signal_length
        tr_signal_stride = self.tr_signal_stride
        tr_signal_length = tr_signal_crop_length + tr_signal_stride
        vl_signal_length = self.crop_signal_length
        vl_signal_stride = vl_signal_length
        tr_sig, tr_gr, tr_op_chn = data_utils.split_into_section(tr_sig, tr_gr, tr_op_chn, tr_signal_length, tr_signal_stride)
        vl_sig, vl_gr, vl_op_chn = data_utils.split_into_section(vl_sig, vl_gr, vl_op_chn, vl_signal_length, vl_signal_stride)
        
        # transformer
        tr_sig_trans = sig_transform.Compose([sig_transform.RandomCrop(tr_signal_crop_length, p=1.0),
                                              sig_transform.RandomFlip(p=0.5),
                                              #sig_transform.RandomGain(gain_limit=(0.9, 1.1), p=0.5),
                                              sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.Cutout(num_holes=int(tr_signal_crop_length*0.1), max_size=1, fill_value=0, p=0.5),
                                              #sig_transform.GaussNoise(std_limit=(0.05, 0.15), mean=0, p=0.5),
                                              sig_transform.AxesTranspose((1,0)),
                                              ]
                                             )
        vl_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        tr_ds = dataset.SignalDataset(tr_sig, tr_gr, tr_op_chn, transform=tr_sig_trans, use_trans_matrix=self.use_trans_matrix)
        vl_ds = dataset.SignalDataset(vl_sig, vl_gr, vl_op_chn, transform=vl_sig_trans, use_trans_matrix=self.use_trans_matrix)

        ## model
        model, use_label = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])
        model = model.cuda()

        ## training
        TR_BATCH_SIZE = 128
        VL_BATCH_SIZE = 16
        if self.sample_weight_per_group is None:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True)
        else:
            tr_loader = dataset.get_dataloader(tr_ds, TR_BATCH_SIZE, shuffle=True, weights=self.sample_weight_per_group[tr_gr])
        vl_loader = dataset.get_dataloader(vl_ds, VL_BATCH_SIZE, shuffle=False)

        LR = 0.001
        #opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        if CP is not None:
            if not FINE_TURNING:
                opt.load_state_dict(CP['optimizer'])
        #tr_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        #vl_criterion = loss.MyCrossEntropyLoss(self.n_label, label_smooth=0.0)
        tr_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[tr_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)
        vl_criterion = loss.ClassBalanced_CrossEntropyLoss(tr_data.iloc[vl_idxs].open_channels.values, self.n_label, beta=0.0, label_smooth=0.1)

        grad_accum_steps = max(128 // TR_BATCH_SIZE, 1)
        start_epoch = 0 if CP is None or FINE_TURNING else CP['epoch']
        EPOCHS = 150

        warmup_epoch=4
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 130, 151], gamma=0.2) #learning rate decay
        if CP is not None:
            if not FINE_TURNING:
                step_scheduler.load_state_dict(CP['scheduler'])

        model = training.train_model(model, tr_loader, vl_loader, use_label,
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head)

        # save
        torch.save(model.state_dict(), self.filename_head + 'model')

        # test
        if run_test:
            self.pred_test(model)

        return

    def pred_test(self, model=None, tg_is_test=True):
        print()
        # model
        if model is None:
            model, _ = self.get_model()
            CP = get_checkpoint(self.checkpoint_path) if self.checkpoint_path is not None else None
            if CP is not None:
                model.load_state_dict(CP['state_dict'])
            model = model.cuda()

        # data
        ts_data = data_utils.add_group(data_utils.get_original_data(is_train=not tg_is_test, cleaned=self.use_cleaned_data), self.ts_unit_length)
        if self.use_trans_matrix:
            if tg_is_test:
                ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_ts_open_channel)        
            else:
                if self.ref_tr_open_channel is None:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, ts_data.open_channels.values)
                else:
                    ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data, self.ref_tr_open_channel)
        else:
            ts_sig, ts_gr, ts_oc  = self.sig_preproc.preprocessing(ts_data)

        # split into section
        ts_signal_length = self.crop_signal_length
        ts_signal_stride = ts_signal_length
        if ts_oc is None:
            ts_sig, ts_gr = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)
        else:
            ts_sig, ts_gr, ts_oc = data_utils.split_into_section(ts_sig, ts_gr, ts_oc, ts_signal_length, ts_signal_stride)

        # transform
        ts_sig_trans = sig_transform.Compose([sig_transform.Normlize(self.sig_preproc.signal_average, self.sig_preproc.signal_std),
                                              sig_transform.AxesTranspose((1,0)),]
                                             )

        # dataset
        ts_ds = dataset.SignalDataset(ts_sig, ts_gr, ts_oc, transform=ts_sig_trans)

        # test
        TS_BATCH_SIZE = 1
        ts_loader = dataset.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        preds = prediction.predict(model, ts_loader)

        if tg_is_test:
            prediction_utils.make_submission(ts_data.time, preds, self.filename_head)
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, None, preds, self.filename_head+'ts_')
        else:
            prediction_utils.make_prediction_result(ts_data.time, ts_data.signal, ts_data.open_channels, preds, self.filename_head+'tr_')
        return

