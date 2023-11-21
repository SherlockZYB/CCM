# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import csv
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from random import random
from torch.optim import lr_scheduler
from torchvision import transforms
from tensorboardX import SummaryWriter
from pymic import TaskType
from pymic.io.nifty_dataset import ClassificationDataset
from pymic.net.net_dict_seg import SegNetDict
from pymic.loss.loss_dict_cls import PyMICClsLossDict
from pymic.net.net_dict_cls import TorchClsNetDict
from pymic.transform.trans_dict import TransformDict
from pymic.transform.view_generator import ContrastiveLearningViewGenerator
from pymic.transform.gaussian_blur import GaussianBlur
from pymic.transform.simclr_crop import SimCLR_RandomResizedCrop
from pymic.transform.pad import Pad
from pymic.io.nifty_dataset import NiftyDataset
from pymic.net_run.agent_abstract import NetRunAgent
from pymic.util.general import mixup, tensor_shape_match
from torch.cuda.amp import GradScaler, autocast
from pymic.loss.seg.mse import MAELoss, MSELoss
from pymic.loss.seg.combined import CombinedLoss
from pymic.loss.seg.deep_sup import DeepSuperviseLoss
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import os
import shutil
import torch
import yaml
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
ReconstructionLossDict = {
    'MAELoss': MAELoss,
    'MSELoss': MSELoss
    }

class BiNet(nn.Module):
    def __init__(self, params):
        super(BiNet, self).__init__()
        net_name  = params['net_type']
        self.net1 = SegNetDict[net_name](params)
        self.net2 = SegNetDict[net_name](params)   

    def forward(self, x):
        feature1, rec_out1 = self.net1(x[:,0])
        feature2, rec_out2 = self.net2(x[:,1])

        if(self.training):
          return feature1, rec_out1, feature2, rec_out2
        else:
          return (rec_out1 + rec_out2) / 2

class ConMAEAgent(NetRunAgent):
    
    def __init__(self, config, stage = 'train'):
        super(ConMAEAgent, self).__init__(config, stage)
        output_act_name = config['network'].get('output_activation', 'sigmoid')
        if(output_act_name == "sigmoid"):
            self.out_act = nn.Sigmoid()
        elif(output_act_name == "tanh"):
            self.out_act = nn.Tanh() 
        else:
            raise ValueError("For reconstruction task, only sigmoid and tanh are " + \
            "supported for output_activation.")
        self.criterion = torch.nn.CrossEntropyLoss().to(torch.device('cuda:0'))
        self.rec_criterion = MAELoss()
        self.temperature = self.config['self_supervised_learning']['temperature']
        self.writer = SummaryWriter()
        self.transform_dict  = TransformDict
        self.loss_weight=self.config['self_supervised_learning']['loss_weight']

    def create_network(self):
        if(self.net is None):
            self.net = BiNet(self.config['network'])
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def get_loss_value(self, data, pred, gt, param = None):
        loss_input_dict = {'prediction':pred, 'ground_truth': gt}
        if(isinstance(pred, tuple) or isinstance(pred, list)):
            device = pred[0].device
        else:
            device = pred.device
        pixel_weight = data.get('pixel_weight', None) 
        if(pixel_weight is not None):
            loss_input_dict['pixel_weight'] = pixel_weight.to(device)

        class_weight = self.config['training'].get('class_weight', None)
        if(class_weight is not None):
            class_num = self.config['network']['class_num']
            assert(len(class_weight) == class_num)
            class_weight = torch.from_numpy(np.asarray(class_weight))
            class_weight = self.convert_tensor_type(class_weight)
            loss_input_dict['class_weight'] = class_weight.to(device)
        loss_value = self.loss_calculator(loss_input_dict)
        return loss_value
    
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset'].get('modal_num', 1)

        transform_key = stage +  '_transform'
        if(stage == "valid" and transform_key not in self.config['dataset']):
            transform_key = "train_transform"
        transform_names = self.config['dataset'][transform_key]
        
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = self.task_type
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        csv_file = self.config['dataset'].get(stage + '_csv', None)
        if(stage == 'test'):
            with_label = False 
        else:
            with_label = self.config['dataset'].get(stage + '_label', True)
        dataset  = NiftyDataset(root_dir  = root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= with_label,
                                transform = data_transform, 
                                task = self.task_type)
        return dataset

    def get_parameters_to_update(self):
        if hasattr(self.net, "get_parameters_to_update"):
            params = self.net.get_parameters_to_update()
        else:
            params = self.net.parameters()
        return params

    def create_loss_calculator(self):
        if(self.loss_dict is None):
            self.loss_dict = ReconstructionLossDict
        loss_name = self.config['training']['loss_type']
        if isinstance(loss_name, (list, tuple)):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
        elif (loss_name not in self.loss_dict):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
        else:
            loss_param = self.config['training']
            loss_param['loss_softmax'] = False
            base_loss = self.loss_dict[loss_name](self.config['training'])
        if(self.config['training'].get('deep_supervise', False)):
            raise ValueError("Deep supervised loss not implemented for reconstruction tasks")
            # weight = self.config['training'].get('deep_supervise_weight', None)
            # mode   = self.config['training'].get('deep_supervise_mode', 2)
            # params = {'deep_supervise_weight': weight, 
            #           'deep_supervise_mode': mode, 
            #           'base_loss':base_loss}
            # self.loss_calculator = DeepSuperviseLoss(params)
        else:
            self.loss_calculator = base_loss

    def get_loss_value(self, data, pred, gt, param = None):
        loss_input_dict = {}
        loss_input_dict['prediction'] = pred
        loss_input_dict['ground_truth'] = gt
        loss_value = self.loss_calculator(loss_input_dict)
        return loss_value
    
    def write_scalars(self, train_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss']}
        acc1_scalar ={'train':train_scalars['acc/top1']}
        acc5_scalar ={'train':train_scalars['acc/top5']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('acc/top1', acc1_scalar, glob_it)
        self.summ_writer.add_scalars('acc/top5', acc5_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        
        logging.info('train loss {0:.4f}, acc/top1 {1:.4f}, acc/top5 {2:.4f} '.format(
            train_scalars['loss'], train_scalars['acc/top1'].item(), train_scalars['acc/top5'].item()) )        
    
    def training(self):
        iter_valid  = self.config['training']['iter_valid']
        train_loss  = 0
        self.net.train()

        scaler = GradScaler(enabled=True)

        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            # get the inputs
            images = data['image']
            inputs = self.convert_tensor_type(images)
            label   = self.convert_tensor_type(data['label'])
            inputs, label = inputs.to(self.device), label.to(self.device)

            with autocast(enabled=True):
                feature1, rec_out1, feature2, rec_out2 = self.net(inputs)
                features = torch.cat([feature1,feature2],0)
                logits, labels = self.info_nce_loss(features)

                label=self.out_act(label)
                rec_out1=self.out_act(rec_out1)
                rec_out2=self.out_act(rec_out2)
                label = torch.squeeze(label,1)
                con_loss = self.criterion(logits, labels)
                rec_loss1 = self.get_loss_value(data, rec_out1, label)
                rec_loss2 = self.get_loss_value(data, rec_out2, label)
                if(self.config['self_supervised_learning']['ablation']=='reconstruction'):
                    rec_loss1=0
                    rec_loss2=0
                elif(self.config['self_supervised_learning']['ablation']=='contrastive'):
                    con_loss=0
                loss = self.loss_weight * con_loss + (rec_loss1+rec_loss2)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()

            train_loss = train_loss + loss.item()
            
            # get dice evaluation for each class
            #! 没有进行warm up
            
            
        top1, top5 = self.accuracy(logits, labels, topk=(1, 5))
        train_avg_loss = train_loss / iter_valid
        train_scalers = {'loss': train_avg_loss, 'acc/top1': top1, 'acc/top5': top5}
        return train_scalers

    def train_valid(self):
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net = nn.DataParallel(self.net, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(self.device)

        ckpt_dir    = self.config['training']['ckpt_save_dir']
        if(ckpt_dir[-1] == "/"):
            ckpt_dir = ckpt_dir[:-1]
        ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
        if(ckpt_prefix is None):
            ckpt_prefix = ckpt_dir.split('/')[-1]
        # iter_start  = self.config['training']['iter_start']       
        iter_start  = 0     
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training'].get('iter_save', None)
        early_stop_it = self.config['training'].get('early_stop_patience', None)
        if(iter_save is None):
            iter_save_list = [iter_max]
        elif(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(0, iter_max + 1, iter_save)

        self.max_val_dice = 100
        self.max_val_it   = 0
        self.best_model_wts = None 
        checkpoint = None
        # initialize the network with pre-trained weights
        ckpt_init_name = self.config['training'].get('ckpt_init_name', None)
        ckpt_init_mode = self.config['training'].get('ckpt_init_mode', 0)
        ckpt_for_optm  = None 
        if(ckpt_init_name is not None):
            checkpoint = torch.load(ckpt_dir + "/" + ckpt_init_name, map_location = self.device)
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = self.net.module.state_dict() if (len(device_ids) > 1) else self.net.state_dict()
            if('net1.' in list(model_dict)[0]):
                if(self.config['training']['ckpt_init_mode']>0):
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
                    k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
                else:
                    pretrained_dict_temp={}
                    for k,v in model_dict.items():
                        if k[5:] in pretrained_dict and tensor_shape_match(pretrained_dict[k[5:]], model_dict[k]):
                            pretrained_dict_temp[k]=pretrained_dict[k[5:]]
                    pretrained_dict=pretrained_dict_temp
                
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
                    k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
            logging.info("Initializing the following parameters with pre-trained model")
            for k in pretrained_dict:
                logging.info(k)
            if (len(device_ids) > 1):
                self.net.module.load_state_dict(pretrained_dict, strict = False)
            else:
                self.net.load_state_dict(pretrained_dict, strict = False)

            if(ckpt_init_mode > 0): # Load  other information
                self.max_val_dice = checkpoint.get('valid_pred', 0)
                iter_start = checkpoint['iteration']
                self.max_val_it = iter_start
                self.best_model_wts = checkpoint['model_state_dict']
                ckpt_for_optm = checkpoint

        self.create_optimizer(self.get_parameters_to_update(), ckpt_for_optm)
        self.create_loss_calculator()

        self.trainIter  = iter(self.train_loader)
        
        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_save_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):
            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            train_scalars = self.training()
            t1 = time.time()
            
            self.scheduler.step()
            
            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training time: {0:.2f}s".format(t1-t0))
            self.write_scalars(train_scalars, lr_value, self.glob_it)

            if(train_scalars['loss'] < self.max_val_dice):
                self.max_val_dice = train_scalars['loss']
                self.max_val_it   = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())
                save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_dice,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_best.pt".format(ckpt_dir, ckpt_prefix)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.max_val_it))
                txt_file.close()

            stop_now = True if (early_stop_it is not None and \
                self.glob_it - self.max_val_it > early_stop_it) else False
            if ((self.glob_it in iter_save_list) or stop_now):
                save_dict = {'iteration': self.glob_it,
                             'valid_pred': train_scalars['acc/top1'],
                             'model_state_dict': self.net.module.state_dict() \
                                 if len(device_ids) > 1 else self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.glob_it))
                txt_file.close()
            if(stop_now):
                logging.info("The training is early stopped")
                break
        # save the best performing checkpoint
        logging.info('The best performing iter is {0:}, valid dice {1:}'.format(\
            self.max_val_it, self.max_val_dice))
        self.summ_writer.close()