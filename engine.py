
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import os

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

import pandas as pd
import numpy as np


#####################################
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    disable_amp: bool = False):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print('debug33: ',samples.shape)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if disable_amp:
            # Disable AMP and try to solve the NaN issue. 
            # Ref: https://github.com/facebookresearch/deit/issues/29
            outputs = model(samples)
            loss = criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if disable_amp:
            loss.backward()
            optimizer.step()
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, disable_amp):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if disable_amp:
            output = model(images)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,3))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate_eval(data_loader, model, device, disable_amp, num_classes, resume, fold, tfold, testing_aug_num, pred_state):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    res_dict = {}
    batch_i = 0
    attn_shape = 0
    for images, target, img_dir_path_ in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        ########<=========
        with torch.no_grad():
            # compute output
            if disable_amp:
                output, feas_, attn_w = model(images)
                loss = criterion(output, target)
            else:
                with torch.cuda.amp.autocast():
                    output, feas_, attn_w = model(images)
                    loss = criterion(output, target)
                    output = F.softmax(output, dim=1)
            acc1, acc5 = accuracy(output, target, topk=(1,3))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            if batch_i==0:
                attn_shape = attn_w.shape
            res_dict[batch_i] = [target.cpu().numpy(), output.cpu().numpy(), feas_.cpu().numpy(), attn_w.cpu().numpy(), img_dir_path_]

        batch_i += 1
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    ################################################
    num_feas_ = 320
    num_attn_w = attn_shape[-1]
    pid_list_tmp = []
    y_true_tmp = []
    y_pred_tmp_dict = {}
    for i in range(num_classes):
        y_pred_tmp_dict['y_pred_' + str(i)] = []
    for i in range(num_feas_):
        y_pred_tmp_dict['fea_' + str(i)] = []
    for i in range(num_attn_w):
        y_pred_tmp_dict['attn_w_' + str(i)] = []
    for k, v in res_dict.items():
        b_pid_list = v[-1]
        b_y_true = v[0]
        b_y_pred = v[1]
        feas_all = v[2]
        b_attn_w = v[3]

        for i in range(len(b_pid_list)):
            pid_list_tmp.append(b_pid_list[i])
            y_true_tmp.append(b_y_true[i])
            for j in range(num_classes):
                y_pred_tmp_dict['y_pred_' + str(j)].append(b_y_pred[i, j])
            for kk in range(num_feas_):
                y_pred_tmp_dict['fea_' + str(kk)].append(feas_all[i, kk])
            for k in range(num_attn_w):
                y_pred_tmp_dict['attn_w_' + str(k)].append(b_attn_w[i, k])

    res_name = resume.split('/')[-3] + '~' + resume.split('/')[-1] + '_' + str(fold).zfill(
        2) + '_' + str(tfold).zfill(2) + '_' + str(testing_aug_num).zfill(2) +'~'+pred_state+'~res.csv'
    total_dict = {'pid': pid_list_tmp, 'y_true': y_true_tmp}
    total_dict.update(y_pred_tmp_dict)
    print(res_name)
    df_res_ = pd.DataFrame(total_dict)
    df_res_.to_csv(res_name)



    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
