# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from pycm import *
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from util.misc import misc_measures

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            visual_features = model.visual(samples)
            #print(visual_features.shape)
            outputs = model.visual.head(visual_features)
            #print(outputs.shape)
            #print('outputs: ',outputs)
            #print('targets: ',targets)
            #outputs = model(samples)
            loss = criterion(outputs, targets)
            if mixup_fn is not None:
                loss = loss.sum()
        
        #loss_scaler.scale(loss).backward()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        #loss_scaler.scale(loss).backward()
        #loss_scaler.step(optimizer)
        #loss_scaler.update()
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            #loss_scaler.step(optimizer)
            #loss_scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, task, mode, num_class, epoch, id_map=None, best_epoch=None, finetune=None, seed=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    path_list = []
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        path = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            #output = model(images)
            if hasattr(model, "visual"):
                visual_features = model.visual(images)
                output = model.visual.head(visual_features)
            else:
                visual_features = model(images)
                output = model.head(visual_features)
            #visual_features = model.visual(images)
            #output = model.visual.head(visual_features)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())
            path_list.extend(path)

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc, class_F1 = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    class_auc_roc = []
    y_true_binary = label_binarize(true_label_decode_list, classes=np.unique(true_label_decode_list))
    for i in range(y_true_binary.shape[1]):
        #print(y_true_binary.shape)
        #print(prediction_list)
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], np.array(prediction_list)[:, i])
        roc_auc = auc(fpr, tpr)
        class_auc_roc.append(roc_auc)

    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro') 
    class_auc_pr = []#average_precision_score(true_label_onehot_list, prediction_list)
    for i in range(y_true_binary.shape[1]):
        #precision, recall, _ = precision_recall_curve(y_true_binary[:, i], prediction_list[:, i])
        avg_precision = average_precision_score(y_true_binary[:, i], np.array(prediction_list)[:, i])
        class_auc_pr.append(avg_precision)
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = os.path.join(task, 'metrics_{}.csv'.format(mode))
    file_empty = False
    epoch_row_index = None
    if not os.path.exists(results_path):
        file_empty = True
    if not file_empty and mode=='val':
        with open(results_path, mode='r', newline='', encoding='utf8') as cfa:
            reader = csv.DictReader(cfa)
            datan = list(reader)
        epoch_row_index = next((index for index, row in enumerate(datan) if row['epoch'] == str(epoch)), None)
        if epoch_row_index is not None:
            new_columns = ['test_acc', 'test_sensitivity', 'test_specificity', 'test_precision', 'test_auc_roc', 'test_auc_pr', 'test_F1', 'test_mcc', 'test_loss']
            for i,column in enumerate(new_columns):
                datan[epoch_row_index][column] = [acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss][i]

    if epoch_row_index is not None:
        with open(results_path, mode='w',newline='',encoding='utf8') as cfa:
            wf = csv.DictWriter(cfa, fieldnames=datan[0].keys())
            wf.writeheader()
            wf.writerows(datan)
    else:
        with open(results_path, mode='a',newline='',encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            if file_empty:
                if best_epoch!=None:
                    header = ['epoch', 'acc', 'sensitivity', 'specificity', 'precision', 'auc_roc', 'auc_pr', 'F1', 'mcc', 'loss', 'model', 'best_epoch']
                else:
                    header = ['epoch', 'acc', 'sensitivity', 'specificity', 'precision', 'auc_roc', 'auc_pr', 'F1', 'mcc', 'loss', 'model']
                wf.writerow(header)
            if best_epoch!=None:
                data2=[[epoch, acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss,finetune, best_epoch]]
            else:
                data2=[[epoch, acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss,finetune]]
            for i in data2:
                wf.writerow(i)
            
    
    if mode=='test':
        out_df = pd.DataFrame()
        out_df['impath'] = path_list
        out_df["true"] = true_label_decode_list
        out_df["pred"] = prediction_decode_list
        out_df['prob'] = np.amax(prediction_list, axis=1)
        #print("prediction_list: ",prediction_list)
        out_df['pid']=out_df['impath'].apply(lambda x:os.path.basename(x).split('_')[0])
        #out_df.to_csv(os.path.join(task, str(epoch)+'_detailed_pred.csv'),index=False)
        if seed != None:
            out_df.to_csv(os.path.join(task, str(epoch)+"_fold_"+str(seed)+'_detailed_pred.csv'),index=False)
        else:
            out_df.to_csv(os.path.join(task, str(epoch)+'_detailed_pred.csv'),index=False)

        class_df = pd.DataFrame()
        if len(class_F1)==len(class_auc_roc) and len(class_F1)==len(class_auc_pr):
            #print('F1: ', class_F1)
            #print('AUROC: ', class_auc_roc)
            class_df['F1']=class_F1
            class_df['AUROC']=class_auc_roc
            class_df['AUCPR']=class_auc_pr
            if seed!=None:
                class_df.to_csv(os.path.join(task, str(epoch)+"_fold_"+str(seed)+'_class_wise_metrics.csv'),index=False)
            else:
                class_df.to_csv(os.path.join(task, str(epoch)+'_class_wise_metrics.csv'),index=False)
            
        #class_df.to_csv(os.path.join(task, str(epoch)+'_class_wise_metrics.csv'),index=False)
        #print('class_f1: ', class_F1)
        #print('class_auroc: ', class_auc_roc)
        #print('class_aucpr: ', class_auc_pr)
        

        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=False,plot_lib="matplotlib")
        if id_map!=None:
            plt.yticks(ticks=range(len(id_map)), labels=list(id_map.keys()), rotation=45, ha="right")
            plt.xticks(ticks=range(len(id_map)), labels=list(id_map.keys()), rotation=45, ha="right")
        #plt.savefig(os.path.join(task, str(epoch)+'_confusion_matrix_test.jpg'),dpi=600,bbox_inches ='tight')
        if seed !=None:
            plt.savefig(os.path.join(task, str(epoch)+"_fold_"+str(seed)+'_confusion_matrix_test.jpg'),dpi=600,bbox_inches ='tight')
        else:
            plt.savefig(os.path.join(task, str(epoch)+'_confusion_matrix_test.jpg'),dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc,F1

