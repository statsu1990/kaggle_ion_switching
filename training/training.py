import torch
import numpy as np
from tqdm import tqdm

from .scheduler import WarmUpLR
from .train_utils import save_log, save_checkpoint
from data import metric

class MacroF1ScoreCalculater:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.group = []

    def add_result(self, y_true, y_pred, group):
        self.time_length = y_true.shape[1]

        self.y_true.append(np.ravel(y_true))
        self.y_pred.append(np.ravel(y_pred))
        self.group.append(np.repeat(group, y_true.shape[1]))

    def calc_score(self, use_group=False):
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        group = np.concatenate(self.group)

        print("All score")
        score = metric.macro_f1(y_true, y_pred, group=None, print_log=True)
        if use_group:
            print("Group score")
            scores_gr = metric.macro_f1(y_true, y_pred, group=group, print_log=True)

        return score

    def calc_score_region_wise(self, num_region):
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)

        region_idxs = (np.arange(len(y_true)) % self.time_length) // (self.time_length // num_region)

        scores = []
        for rg in range(num_region):
            #print(np.sum(region_idxs==rg))
            scores.append(metric.macro_f1(y_true[region_idxs==rg], y_pred[region_idxs==rg], group=None, print_log=False))

        print([str(s)[:5] for s in scores])

        return scores

def real_to_integer(value, min=0, max=10):
    return np.clip(np.round(value).astype('int'), min, max)

def trainer(net, loader, criterion, optimizer, grad_accum_steps, warmup_scheduler, use_label=False, classification=True, score_use_group=False):
    net.train()

    total_loss = 0
    score_calclater = MacroF1ScoreCalculater()

    optimizer.zero_grad()
    #for batch_idx, (signals, group, open_channels, trams_mtrxs) in enumerate(tqdm(loader)):
    for batch_idx, data in enumerate(tqdm(loader)):

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        if len(data) == 3:
            signals, group, open_channels = data[0].cuda(), data[1].cuda(), data[2].cuda()
        else:
            signals, group, open_channels, trams_mtrxs = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

        if use_label:
            outputs, additional_output = net(signals, trams_mtrxs, open_channels)
            loss = criterion(outputs, additional_output[0], additional_output[1], additional_output[2])
        else:
            if len(data) == 3:
                outputs = net(signals)
            else:
                outputs = net(signals, trams_mtrxs)
            loss = criterion(outputs, open_channels)

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # loss
        total_loss += loss.item() * grad_accum_steps
        
        # score
        with torch.no_grad():
            if classification:
                score_calclater.add_result(open_channels.cpu().numpy(), outputs.max(2)[1].cpu().numpy(), group.cpu().numpy())
            else:
                # regression
                score_calclater.add_result(open_channels.cpu().numpy(), real_to_integer(outputs.cpu().numpy()), group.cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)

    # score
    score = score_calclater.calc_score(use_group=score_use_group)
    score_calclater.calc_score_region_wise(num_region=10)

    print('Train Loss: %.4f | Score: %.4f' % (total_loss, score))
    
    return total_loss, score

def tester(net, loader, criterion, classification=True, score_use_group=False):
    net.eval()

    total_loss = 0
    score_calclater = MacroF1ScoreCalculater()

    with torch.no_grad():
        #for batch_idx, (signals, group, open_channels, trams_mtrxs) in enumerate(tqdm(loader)):
        for batch_idx, data in enumerate(tqdm(loader)):
            if len(data) == 3:
                signals, group, open_channels = data[0].cuda(), data[1].cuda(), data[2].cuda()
            else:
                signals, group, open_channels, trams_mtrxs = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            if len(data) == 3:
                outputs = net(signals)
            else:
                outputs = net(signals, trams_mtrxs)
            if criterion is not None:
                loss = criterion(outputs, open_channels)
                # loss
                total_loss += loss.item()
            else:
                total_loss += 0
        
            # score
            if classification:
                score_calclater.add_result(open_channels.cpu().numpy(), outputs.max(2)[1].cpu().numpy(), group.cpu().numpy())
            else:
                # regression
                score_calclater.add_result(open_channels.cpu().numpy(), real_to_integer(outputs.cpu().numpy()), group.cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)

    # score
    score = score_calclater.calc_score(use_group=score_use_group)
    score_calclater.calc_score_region_wise(num_region=10)

    print('Valid Loss: %.4f | Score: %.4f' % (total_loss, score))
    
    return total_loss, score

def train_model(net, tr_loader, vl_loader, use_label, 
                optimizer, tr_criterion, vl_criterion, 
                grad_accum_steps, start_epoch, epochs, 
                warmup_epoch, step_scheduler, filename_head='', 
                classification=True):

    # warmup_scheduler
    if start_epoch < warmup_epoch:
        warmup_scheduler = WarmUpLR(optimizer, len(tr_loader) * warmup_epoch)
    else:
        warmup_scheduler = None
    
    # train
    loglist = []
    for epoch in range(start_epoch, epochs):
        if epoch > warmup_epoch - 1:
            warm_sch = None
            step_scheduler.step()
        else:
            warm_sch = warmup_scheduler

        print('\nepoch ', epoch)
        for param_group in optimizer.param_groups:
            print('lr ', param_group['lr'])
            now_lr = param_group['lr']

        score_use_group = (epoch == epochs - 1)

        tr_log = trainer(net, tr_loader, tr_criterion, optimizer, grad_accum_steps, warm_sch, use_label, classification, score_use_group)
        vl_log = tester(net, vl_loader, vl_criterion, classification, score_use_group)

        # save checkpoint
        save_checkpoint(epoch, net, optimizer, step_scheduler, filename_head + 'checkpoint')

        # save log
        loglist.append([epoch] + [now_lr] + list(tr_log) + list(vl_log))
        colmuns = ['epoch', 'lr', 'tr_loss', 'tr_score', 'vl_loss', 'vl_score']
        save_log(loglist, colmuns, filename_head + 'training_log.csv')

    return net

def trainer_v2(net, loader, criterion, optimizer, grad_accum_steps, warmup_scheduler, use_label=False, classification=True, score_use_group=False):
    net.train()

    total_loss = 0
    score_calclater = MacroF1ScoreCalculater()

    optimizer.zero_grad()
    #for batch_idx, (signals, group, open_channels, trams_mtrxs) in enumerate(tqdm(loader)):
    for batch_idx, data in enumerate(tqdm(loader)):

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        if len(data) == 3:
            signals, group, open_channels = data[0].cuda(), data[1].cuda(), data[2].cuda()
        else:
            signals, group, open_channels, trams_mtrxs = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

        #
        cor_label = torch.eq(torch.argmax(signals, dim=1), open_channels).long()

        if len(data) == 3:
            outputs = net(signals)
        else:
            outputs = net(signals, trams_mtrxs)
        loss = criterion(outputs, cor_label)

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # loss
        total_loss += loss.item() * grad_accum_steps
        
        # score
        with torch.no_grad():
            sig_arg = np.argsort(signals.cpu().numpy(), axis=1)
            pred = outputs.max(2)[1].cpu().numpy()
            pred_label = sig_arg[:,-1,:]
            pred_label[pred==0] = (sig_arg[:,-2,:])[pred==0]

            score_calclater.add_result(open_channels.cpu().numpy(), pred_label, group.cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)

    # score
    score = score_calclater.calc_score(use_group=score_use_group)
    score_calclater.calc_score_region_wise(num_region=10)

    print('Train Loss: %.4f | Score: %.4f' % (total_loss, score))
    
    return total_loss, score

def tester_v2(net, loader, criterion, classification=True, score_use_group=False):
    net.eval()

    total_loss = 0
    score_calclater = MacroF1ScoreCalculater()

    with torch.no_grad():
        #for batch_idx, (signals, group, open_channels, trams_mtrxs) in enumerate(tqdm(loader)):
        for batch_idx, data in enumerate(tqdm(loader)):
            if len(data) == 3:
                signals, group, open_channels = data[0].cuda(), data[1].cuda(), data[2].cuda()
            else:
                signals, group, open_channels, trams_mtrxs = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            #
            cor_label = torch.eq(torch.argmax(signals, dim=1), open_channels).long()

            if len(data) == 3:
                outputs = net(signals)
            else:
                outputs = net(signals, trams_mtrxs)
            if criterion is not None:
                loss = criterion(outputs, cor_label)
                # loss
                total_loss += loss.item()
            else:
                total_loss += 0
        
            # score
            sig_arg = np.argsort(signals.cpu().numpy(), axis=1)
            pred = outputs.max(2)[1].cpu().numpy()
            pred_label = sig_arg[:,-1,:]
            pred_label[pred==0] = (sig_arg[:,-2,:])[pred==0]
            score_calclater.add_result(open_channels.cpu().numpy(), pred_label, group.cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)

    # score
    score = score_calclater.calc_score(use_group=score_use_group)
    score_calclater.calc_score_region_wise(num_region=10)

    print('Valid Loss: %.4f | Score: %.4f' % (total_loss, score))
    
    return total_loss, score

def train_model_v2(net, tr_loader, vl_loader, use_label, 
                optimizer, tr_criterion, vl_criterion, 
                grad_accum_steps, start_epoch, epochs, 
                warmup_epoch, step_scheduler, filename_head='', 
                classification=True):

    # warmup_scheduler
    if start_epoch < warmup_epoch:
        warmup_scheduler = WarmUpLR(optimizer, len(tr_loader) * warmup_epoch)
    else:
        warmup_scheduler = None
    
    # train
    loglist = []
    for epoch in range(start_epoch, epochs):
        if epoch > warmup_epoch - 1:
            warm_sch = None
            step_scheduler.step()
        else:
            warm_sch = warmup_scheduler

        print('\nepoch ', epoch)
        for param_group in optimizer.param_groups:
            print('lr ', param_group['lr'])
            now_lr = param_group['lr']

        score_use_group = (epoch == epochs - 1)

        tr_log = trainer_v2(net, tr_loader, tr_criterion, optimizer, grad_accum_steps, warm_sch, use_label, classification, score_use_group)
        vl_log = tester_v2(net, vl_loader, vl_criterion, classification, score_use_group)

        # save checkpoint
        save_checkpoint(epoch, net, optimizer, step_scheduler, filename_head + 'checkpoint')

        # save log
        loglist.append([epoch] + [now_lr] + list(tr_log) + list(vl_log))
        colmuns = ['epoch', 'lr', 'tr_loss', 'tr_score', 'vl_loss', 'vl_score']
        save_log(loglist, colmuns, filename_head + 'training_log.csv')

    return net
