import os


import argparse
import datetime
import numpy as np
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pywt import dwt2, idwt2
import matplotlib.pyplot as plt

from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_eval
from samplers import RASampler
# import models.coat
import utils
from args import get_args_parser
from aug_classfication_mil import aug_process


###########################################################################################
def pading_fix(img, is_padding=True, mode='train', im_pad_val=0, pad_shape=None):
    img = img[np.newaxis,:,:]
    if is_padding:
        num_channel,height, width = img.shape
        if pad_shape==None:
            max_edge = np.max([height, width])
        else:
            max_edge = pad_shape
        new_img = np.zeros((1, max_edge, max_edge)) + im_pad_val

        if mode == 'train':
            random_bias_c_l = 0
            random_bias_c_r = 0

            random_bias_h_l = int(np.random.randint(max_edge - height) if (max_edge-height)>0 else 0)
            random_bias_h_r = int(max_edge-height-random_bias_h_l)

            random_bias_w_l = int(np.random.randint(max_edge - width) if (max_edge-width)>0 else 0)
            random_bias_w_r = int(max_edge-width-random_bias_w_l)

            new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r),\
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
        else:
            random_bias_c_l = 0
            random_bias_c_r = 0

            random_bias_h_l = 0
            random_bias_h_r = int(max_edge - height - random_bias_h_l)

            random_bias_w_l = 0
            random_bias_w_r = int(max_edge - width - random_bias_w_l)
            new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r), \
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
        return new_img[0]
    else:
        return img[0]

def center_pading_fix(img, is_padding=True, im_pad_val=0, pad_shape=None):
    img = img[np.newaxis,:,:]
    if is_padding:
        num_channel,height, width = img.shape
        if pad_shape==None:
            max_edge = np.max([height, width])
        else:
            max_edge = pad_shape
        new_img = np.zeros((1, max_edge, max_edge)) + im_pad_val

        random_bias_c_l = 0
        random_bias_c_r = 0

        random_bias_h_l = int((max_edge - height)//2 if (max_edge-height)>0 else 0)
        random_bias_h_r = int(max_edge-height-random_bias_h_l)

        random_bias_w_l = int((max_edge - width)//2 if (max_edge-width)>0 else 0)
        random_bias_w_r = int(max_edge-width-random_bias_w_l)

        new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r),\
                                      (random_bias_h_l, random_bias_h_r),
                                      (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
        return new_img[0]
    else:
        return img[0]


def random_crop_to_fix_shape(img, fix_shape=(64,64)):
    h,w = img.shape
    x = np.random.randint(0, h-fix_shape[0])
    y = np.random.randint(0, w-fix_shape[1])
    box = (x, y, x+fix_shape[0], y+fix_shape[1])
    return box

def compute_iou(box_1, box_2):
    xmin = np.max([box_1[0], box_2[0]])
    ymin = np.max([box_1[1], box_2[1]])
    xmax = np.min([box_1[2], box_2[2]])
    ymax = np.min([box_1[3], box_2[3]])
    insection = (xmax-xmin)*(ymax-ymin)
    area_1 = (box_1[2]-box_1[0])*(box_1[3]-box_1[1])
    area_2 = (box_2[2]-box_2[0])*(box_2[3]-box_2[1])
    iou = insection/(area_1+area_2-insection+1e-6)
    return iou



def product_boxes_valid(img, fix_shape=(64,64), iou_thresh=0.3):   #0.3

    if img.shape[0]<=fix_shape[0]:
        box_tmp = [0,0, fix_shape[0], fix_shape[1]]
        return [box_tmp, box_tmp]
    box_1 = crop_center_v2(img, fix_shape[0], fix_shape[1])

    while 1:

        box_2 = random_crop_to_fix_shape(img, fix_shape=fix_shape)
        iou_ = compute_iou(box_1, box_2)
        if iou_<iou_thresh:

            return [box_1, box_2]



def crop_center_v2(img, croph, cropw):
    height, width = img.shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return [starth, startw, starth+croph, startw+cropw]

def get_sample_v2(img_, is_padding=True, mode='train', im_pad_val=-1024, img_size=64):
    img_ = center_pading_fix(img_, is_padding=True, im_pad_val=im_pad_val)
    if img_.shape[1]<img_size:
        img_ = pading_fix(img_, is_padding=True, mode=mode, im_pad_val=im_pad_val, pad_shape=img_size)
    if mode == 'train':
        box_1, box_2 = product_boxes_valid(img_, fix_shape=(img_size, img_size))
        img_1 = img_[box_1[0]:box_1[2], box_1[1]:box_1[3]]
        img_2 = img_[box_2[0]:box_2[2], box_2[1]:box_2[3]]
    else:
        box_1, box_2 = product_boxes_valid(img_, fix_shape=(img_size, img_size))
        img_1 = img_[box_1[0]:box_1[2], box_1[1]:box_1[3]]
        img_2 = img_[box_2[0]:box_2[2], box_2[1]:box_2[3]]
    img_new = np.zeros((2, 1, img_size, img_size))
    img_new[0,0] = img_1
    img_new[1,0] = img_2
    return img_new

# @jit
def aug_pipe(img, has_imgaug=True, random_rotate=True, random_lightness=True, random_transpose=True, im_pad_val=0):
    h_, w_ = img.shape
    img_ = np.zeros((1, h_, w_), dtype=np.float32)
    img_[0] = img
    if has_imgaug:
        img_ = img_.transpose((2,1,0))
        img_ = aug_process(img_, im_pad_val=im_pad_val)
        img_ = img_.transpose((2,1,0))

    return img_[0]

def slice_sample(num_img_, num_instance):

    s_ids_ = list(np.linspace(0, num_img_, num_instance+1))

    st_id_list = []
    # print('^^^^^^', s_ids_)
    for i in range(1, len(s_ids_)):
        start_ = int(s_ids_[i-1])
        stop_ = int(s_ids_[i])
        if s_ids_[i-1]-start_>0:
            start_ += 1
        if s_ids_[i]-stop_>0:
            stop_ += 1
        st_list = list(range(start_, stop_))
        np.random.shuffle(st_list)
        # print('++++++here: ', st_list)
        st_id_list.append(st_list[0])
    assert len(set(st_id_list))==num_instance, 'error in slice sample'
    return st_id_list

class MRIDataset_2p5D(Dataset):
    def __init__(self, index_list, data_shape=(1, 64, 64), label_name='label', idx_name='pid', mode='train', csv_path='',
                 reverse=False, has_aug=False, im_pad_val=0, extend_num=0, num_instance_list=[3, 3, 3],
                 data_root_dir='~/dataset', slice_prefix='ImageFileName'):
        super(MRIDataset_2p5D, self).__init__()
        self.mode = mode
        self.idx_name = idx_name
        self.data_root_dir = data_root_dir
        self.slice_prefix = slice_prefix
        self.data_shape = data_shape
        self.reverse = reverse
        self.label_name = label_name
        self.index_list = index_list
        self.add_gaussian_mask = False
        self.add_edge = False
        self.detail_enhancement = True
        self.wavelet_trans = False
        self.padding = True
        self.resize = True
        self.mosaic = False
        self.has_aug = has_aug
        self.random_rotate = True
        self.random_lightness = True
        self.random_transpose = True
        self.random_mirror = True
        self.random_brightness = False
        self.random_gaussian_noise = False
        self.random_rician_noise = False
        self.im_pad_val = im_pad_val
        self.extend_num = extend_num
        self.len = len(index_list)
        self.all_df = pd.read_csv(csv_path)
        self.num_instance_list = num_instance_list
        print('=== mode:' + self.mode)
        print('=== num of samples: ', self.len)

    def sample_generator(self, img_dir_path):

        slice_name_list = os.listdir(img_dir_path)
        slice_name_list = sorted(slice_name_list)
        img_path_list = [os.path.join(img_dir_path, item) for item in slice_name_list]
        ori_num_img = len(img_path_list)

        if ori_num_img<=self.num_instance_list[0]:
            num_suppl = int((self.num_instance_list[0]-ori_num_img+2)/2)
            new_img_path_list = ['padding_path']*num_suppl+img_path_list[1:-1]+['padding_path']*(self.num_instance_list[0]-ori_num_img+2-num_suppl)
            new_img_path_list = new_img_path_list[1:-1]
            assert len(new_img_path_list) == (self.num_instance_list[0] - 2), 'error 2: ' +str(len(new_img_path_list))+' '+str(num_suppl)+' '+str(ori_num_img)
        else:
            axis_c_list = slice_sample(ori_num_img, self.num_instance_list[0])
            # print(len(axis_c_list), ori_num_img)
            new_img_path_list = []
            for i in axis_c_list[1:-1]:
                new_img_path_list.append(img_path_list[i])
        assert len(new_img_path_list)==(self.num_instance_list[0]-2), 'error 2: '+str(len(new_img_path_list))+' '+str(self.num_instance_list[0]-2)
        instance_list = []
        for i in range(self.num_instance_list[0]-2):
            if new_img_path_list[i]=='padding_path':
                img_ = np.zeros((self.data_shape[2], self.data_shape[2]))
            else:
                img_ = np.load(new_img_path_list[i]) # z,x,y

            if self.mode == 'train':
                if self.random_lightness:
                    # random lightness augmentation
                    randint = np.random.uniform(low=-0.1, high=0.1)
                    randint = np.round(randint)
                    img_ = img_ + randint

                if np.random.random() > 0.5:
                    img_ = np.transpose(img_, (1, 0))

                # 随机旋转
                num_rot = (0, 1, 2, 3)
                num_rot = np.random.choice(num_rot)
                if num_rot > 0 and np.random.random() > 0.5:
                    axes_ = (0, 1)
                    img_ = np.rot90(img_, num_rot, axes=axes_)
                if self.has_aug:
                    img_ = aug_pipe(img_)

            img_new = get_sample_v2(img_, is_padding=True, mode=self.mode, im_pad_val=self.im_pad_val,
                       img_size=self.data_shape[2])   ### shape=(2,1,img_size,img_size)
            instance_list.append(img_new)
        num_instance = len(instance_list)
        sample = np.zeros((num_instance*2, self.data_shape[0], self.data_shape[1], self.data_shape[2]), dtype=np.float32)
        for i in range(num_instance):
            sample[2*i:2*(i+1),:,:,:] = instance_list[i]
        return sample

    def __getitem__(self, index):
        img_dir_path_ = self.index_list[index]
        target = np.array(self.all_df[self.all_df['image_dir_path'] == img_dir_path_][self.label_name])[0]
        target = int(target)

        sample = self.sample_generator(img_dir_path_)
        if self.mode=='eval':
            return sample, target, img_dir_path_
        return torch.from_numpy(sample).type(torch.FloatTensor), target  # torch.from_numpy(target).long()

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_split_deterministic(all_keys, fold=0, num_splits=4, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """

    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys

def func_1130_1(df_tmp, ids_list):
    img_dir_path_list = []
    for i in range(len(ids_list)):
        case_pid_tmp_ = ids_list[i]
        img_dir_path_list_tmp = df_tmp[df_tmp['case_pid']==case_pid_tmp_]['image_dir_path'].tolist()
        img_dir_path_list+=img_dir_path_list_tmp
    return img_dir_path_list

def main(args):
    utils.init_distributed_mode(args)

    # Debug mode.
    if args.debug:
        import debugpy
        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")
        
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    ################################################################
    csvPath = args.data_path
    train_fold_idx = args.fold
    tfold = args.tfold
    fix_data_shape = args.crop_size
    ind_name = 'case_pid'

    np.random.seed(12345)
    df_tmp = pd.read_csv(csvPath)

    patients0 = np.array(df_tmp[(df_tmp['label'] == 0)][ind_name].unique())
    patients1 = np.array(df_tmp[(df_tmp['label'] == 1)][ind_name].unique())
    patients2 = np.array(df_tmp[(df_tmp['label'] == 2)][ind_name].unique())

    train_idx00, test_idx0 = get_split_deterministic(patients0, fold=tfold, num_splits=5)
    train_idx11, test_idx1 = get_split_deterministic(patients1, fold=tfold, num_splits=5)
    train_idx22, test_idx2 = get_split_deterministic(patients2, fold=tfold, num_splits=5)


    train_idx0, valid_idx0 = get_split_deterministic(train_idx00, fold=train_fold_idx, num_splits=4)
    train_idx1, valid_idx1 = get_split_deterministic(train_idx11, fold=train_fold_idx, num_splits=4)
    train_idx2, valid_idx2 = get_split_deterministic(train_idx22, fold=train_fold_idx, num_splits=4)

    train_idx = np.concatenate([train_idx0, train_idx1, train_idx2], axis=0)
    np.random.shuffle(train_idx)

    valid_idx = np.concatenate([valid_idx0, valid_idx1, valid_idx2], axis=0)
    np.random.shuffle(valid_idx)

    test_idx = np.concatenate([test_idx0, test_idx1, test_idx2], axis=0)
    np.random.shuffle(test_idx)

    train_sids = func_1130_1(df_tmp, train_idx)
    np.random.shuffle(train_sids)
    valid_sids = func_1130_1(df_tmp, valid_idx)
    np.random.shuffle(valid_sids)
    test_sids = func_1130_1(df_tmp, test_idx)
    np.random.shuffle(test_sids)


    num_instance_list_tmp = [args.num_instance_c]
    valid_mode = 'valid'
    index_list_tmp_ = valid_sids
    if args.eval:
        valid_mode = 'eval'
        if args.pred_state=='train':
            index_list_tmp_ = train_sids
        elif args.pred_state=='valid':
            index_list_tmp_ = valid_sids
        else:
            index_list_tmp_ = test_sids
        if args.testing_aug_num:
            index_list_tmp_ = list(index_list_tmp_) * int(args.testing_aug_num)
    no_draw_cam = True
    dataset_train = MRIDataset_2p5D(index_list=train_sids,
                                    data_shape=(args.in_chans, fix_data_shape, fix_data_shape),
                                    label_name='label',
                                    idx_name=ind_name,
                                    mode='train',
                                    csv_path=csvPath,
                                    reverse=False,
                                    has_aug=args.has_aug,
                                    extend_num=args.extend_num,
                                    num_instance_list=num_instance_list_tmp,
                            )
    if no_draw_cam:
        dataset_val = MRIDataset_2p5D(index_list=index_list_tmp_,
                                        data_shape=(args.in_chans, fix_data_shape, fix_data_shape),
                                        label_name='label',
                                        idx_name=ind_name,
                                        mode=valid_mode,
                                        csv_path=csvPath,
                                        reverse=False,
                                        has_aug=False,
                                        extend_num=args.extend_num,
                                        num_instance_list=num_instance_list_tmp,
                              )

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem, 
        drop_last=True,)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=int(1.5 * args.batch_size), 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem, 
        drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model, 
        pretrained=True,
        num_classes=args.num_classes,
        drop_rate=args.drop, 
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        in_chans=args.in_chans,
        img_size=fix_data_shape,
        final_drop=args.final_drop,
        mode='eval' if args.eval else 'train',
        **eval(args.model_kwargs))
    # print(model)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model, 
            decay=args.model_ema_decay, 
            device='cpu' if args.model_ema_force_cpu else '', 
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)
    print('lr: ', args.lr)
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()


    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()

    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            raise NotImplementedError
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    if args.eval:
        test_stats = evaluate_eval(data_loader_val, model, device, disable_amp=args.disable_amp,
                                   num_classes=args.num_classes,
                                   resume=args.resume, fold=args.fold, tfold=args.tfold,
                                   testing_aug_num=args.testing_aug_num, pred_state=args.pred_state)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log.txt").open("a") as f:
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'n_parameters': n_parameters}
                f.write(json.dumps(log_stats) + "\n")
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    # Initial checkpoint saving.
    if args.output_dir:
        checkpoint_paths = [output_dir / 'checkpoints/checkpoint.pth']
        # print(checkpoint_paths)
        if not os.path.exists(args.output_dir+'/checkpoints'):
            os.makedirs(args.output_dir+'/checkpoints')
        for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': -1,  # Note: -1 means initial checkpoint.
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
                                        args.clip_grad, model_ema, mixup_fn, disable_amp=args.disable_amp)

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoints/checkpoint.pth']
            if epoch % args.save_freq == args.save_freq - 1:
                checkpoint_paths.append(output_dir / f'checkpoints/checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)

        if epoch % args.save_freq == args.save_freq - 1:
            test_stats = evaluate(data_loader_val, model, device, disable_amp=args.disable_amp)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                         'epoch': epoch, 
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
