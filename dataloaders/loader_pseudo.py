""" Data Loader for Generating Tasks

Author: Zhao Na, 2020
"""
import os
import random
import math
# import glob
import numpy as np
# import h5py as h5
import transforms3d
# from itertools import  combinations

import torch
from torch.utils.data import Dataset
# import pickle

def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                      base_cls, novel_cls, support=False, random_sample=False, 
                      use_all_classes=False, bg_data_path=None, up_bnd=False, pseudo2novel=None):
    '''
    Args:
        data_path:
        num_point:
        pc_attribs:
        pc_augm:
        pc_augm_config:
        scan_name:
        sampled_classes: # base_cls
        sampled_class:
        support:
        random_sample:
        use_all_classes: in base stage of mpti (pretrain+meta-train), we keep a 'bg' class and set its idx 0. In novel stage and testing stage, there is no reservation of 'bg'!

    Returns:

    '''
    # sampled_classes = list(sampled_classes) # pre-train: train classes.  train: 2-way classes name
    # # # make sure sampled_classes follows order
    # # sampled_classes = sorted(sampled_classes)

    if bg_data_path is not None:
        data = np.load(os.path.join(data_path, bg_data_path, '%s.npy' %scan_name))
    else:
        data = np.load(os.path.join(data_path, 'data', '%s.npy' %scan_name))
    N = data.shape[0] #number of points in this scan (BLOACK)

    if random_sample:
        sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point)) # random sample 2048 points in this block
    else:
        pass

    data = data[sampled_point_inds]
    xyz = data[:, 0:3]
    rgb = data[:, 3:6]


    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)
    if 'XYZ' in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb/255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1) # (2048, 9)

    # get labels
    labels = data[:, 6].astype(np.int)
        
    groundtruth = np.zeros_like(labels) # labels that only availabel in the given classes
    for i, b_cls in enumerate(base_cls):
        groundtruth[labels==b_cls] = i+1
    
    # if bg_data_path is not None:
    #     pseudo_label = data[:, 7]
    # else:
    #     pseudo_label = np.zeros(data.shape[0], dtype=data.dtype)
    if up_bnd:
        pseudo_label = np.zeros_like(labels)
        for i, n_cls in enumerate(novel_cls):
            pseudo_label[labels==n_cls] = i+1
    else:
        fuse_label = data[:, 7].astype(np.int)
        # print(np.unique(fuse_label))
        pseudo_label = np.zeros_like(labels)
        for i, n_cls in enumerate(pseudo2novel):
            pseudo_label[fuse_label==i] = n_cls+1

    pseudo_mask = pseudo_label>0
    gt_bg_mask = groundtruth==0
    pseudo_mask = pseudo_mask & gt_bg_mask
    groundtruth[pseudo_mask] = pseudo_label[pseudo_mask] + len(base_cls)

    return ptcloud, groundtruth

def augment_pointcloud(P, pc_augm_config):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_config['rot'] == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if pc_augm_config['mirror_prob'] > 0:  # mirroring x&y, not z
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)

    if pc_augm_config['jitter']:
        sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P


class TrainDataset(Dataset):
    def __init__(self, data_path, base_cls, novel_cls, class2scans, pseudo_class2scans, num_point=4096, pc_attribs='xyz',
                       pc_augm=False, pc_augm_config=None, bg_data_path=None, up_bnd=False, pseudo2novel=None):
        ''' dataset of base classes
        Args:
            data_path:
            classes: make sure they are in order.
            class2scans:
            mode:
            num_point:
            pc_attribs:
            pc_augm:
            pc_augm_config:
        '''
        super(TrainDataset).__init__()
        self.data_path = data_path # train dataset
        self.base_cls = base_cls # train classes name
        self.novel_cls = novel_cls
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs # xyzrgbXYZ
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        self.bg_data_path = bg_data_path
        self.up_bnd = up_bnd
        self.pseudo2novel = pseudo2novel
        print('bg_data_path',bg_data_path)

        if up_bnd:
            base_block_names = []
            novel_block_names = []
            for k, v in sorted(class2scans.items()):
                if k in base_cls:
                    base_block_names.extend(v)
                if k in novel_cls:
                    novel_block_names.extend(v)
        else:
            base_block_names = []
            novel_block_names = []
            for k, v in sorted(class2scans.items()):
                if k in base_cls:
                    base_block_names.extend(v)
            for k, v in sorted(pseudo_class2scans.items()):
                if k in range(len(novel_cls)):
                    novel_block_names.extend(v)

        base_block_names = list(set(base_block_names))
        novel_block_names = novel_block_names
        self.block_names = base_block_names + novel_block_names
        self.block_names = sorted(self.block_names)
        print('num of base blocks,', len(base_block_names))
        print('num of novel blocks,', len(novel_block_names))

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        block_name = self.block_names[index]
        # print(block_name)
        ptcloud, label = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                                                    self.pc_augm_config, block_name, self.base_cls, self.novel_cls, random_sample=True, 
                                                                    bg_data_path=self.bg_data_path, up_bnd=self.up_bnd, pseudo2novel=self.pseudo2novel)

        return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64)) #, torch.from_numpy(segment_label.astype(np.float32))


if __name__=='__main__':
    from s3dis import S3DISDataset
    cvfold = 0
    data_path = 'datasets/S3DIS/blocks_bs1_s1'
    bg_data_path = 'base0_mix'
    base_cls = [0, 1, 2, 6, 8, 10, 12]
    novel_cls = [3, 4, 5, 7, 9, 11]
    all_class_names = np.arange(13)
    DATASET = S3DISDataset(cvfold, data_path, bg_data_path)
    pseudo2novel = [3,2,1,0,5,4]
    class2scans = {c: DATASET.class2scans[c] for c in all_class_names}
    pseudo_class2scans = {c: DATASET.pseudo_class2scans[c] for c in range(len(pseudo2novel))}
    dataset = TrainDataset(data_path, base_cls, novel_cls, class2scans, pseudo_class2scans, bg_data_path=bg_data_path, pseudo2novel=pseudo2novel)
    for i in range(len(dataset)):
        x, y = dataset[i]
        # print(y.unique())
    # breakpoint()

if False:
    ### pseudo from 2D foundation models
    def sample_pointcloud_pseudo(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                                    sampled_classes, sampled_class=0, support=False, random_sample=False, 
                                    use_all_classes=False, bg_data_path=None):
        
        sampled_classes = list(sampled_classes) # pre-train: train classes.  train: 2-way classes name
        num_class = len(sampled_classes)
        # # make sure sampled_classes follows order
        # sampled_classes = sorted(sampled_classes)

        if bg_data_path is not None:
            data = np.load(os.path.join(data_path, bg_data_path, '%s.npy' %scan_name))
        else:
            data = np.load(os.path.join(data_path, 'data', '%s.npy' %scan_name))
        N = data.shape[0] #number of points in this scan (BLOACK)

        if random_sample:
            sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point)) # random sample 2048 points in this block
        else:
            # If this point cloud is for support/query set, make sure that the sampled points contain target class
            valid_point_inds = np.nonzero(data[:,6] == sampled_class)[0]  # indices of points belonging to the sampled class

            if N < num_point:
                sampled_valid_point_num = len(valid_point_inds)
            else:
                valid_ratio = len(valid_point_inds)/float(N)
                sampled_valid_point_num = int(valid_ratio*num_point)

            sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
            sampled_other_point_inds = np.random.choice(np.arange(N), num_point-sampled_valid_point_num,
                                                        replace=(N<num_point))
            sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])

        data = data[sampled_point_inds]
        xyz = data[:, 0:3]
        rgb = data[:, 3:6]

        xyz_min = np.amin(xyz, axis=0)
        xyz -= xyz_min
        if pc_augm:
            xyz = augment_pointcloud(xyz, pc_augm_config)
        if 'XYZ' in pc_attribs:
            xyz_min = np.amin(xyz, axis=0)
            XYZ = xyz - xyz_min
            xyz_max = np.amax(XYZ, axis=0)
            XYZ = XYZ/xyz_max

        ptcloud = []
        if 'xyz' in pc_attribs: ptcloud.append(xyz)
        if 'rgb' in pc_attribs: ptcloud.append(rgb/255.)
        if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
        ptcloud = np.concatenate(ptcloud, axis=1) # (2048, 9)

        # get labels
        labels = data[:, 6].astype(np.int)
        if use_all_classes == False:
            if support:
                groundtruth = labels==sampled_class # binary label
            else:
                groundtruth = np.zeros_like(labels) # labels that only availabel in the given classes
                for i, label in enumerate(labels):
                    if label in sampled_classes:
                        groundtruth[i] = sampled_classes.index(label)+1
        else:
            if support:
                groundtruth = labels==sampled_class # binary label
            else:
                groundtruth = np.zeros_like(labels) # labels that only availabel in the given classes
                for i, label in enumerate(labels):
                    if label in sampled_classes:
                        groundtruth[i] = sampled_classes.index(label) # no reservation of 'bg' class
                assert groundtruth.max() <= max(sampled_classes)

        if bg_data_path is not None:
            pseudo_label = data[:, 7]
            gt_bg_mask = groundtruth==0
            pseudo_mask = pseudo_label>-1
            pseudo_mask = pseudo_mask & gt_bg_mask
            groundtruth[pseudo_mask] = pseudo_label[pseudo_mask] + num_class + 1
        else:
            print('error!!!')
            # pseudo_label = np.zeros(data.shape[0], dtype=data.dtype)

        return ptcloud, groundtruth #, pseudo_label #segment_label

    class PseudoDataset(Dataset):
        def __init__(self, data_path, classes, class2scans, mode='train', num_point=4096, pc_attribs='xyz',
                        pc_augm=False, pc_augm_config=None, bg_data_path=None):
            ''' dataset of base classes
            Args:
                data_path:
                classes: make sure they are in order.
                class2scans:
                mode:
                num_point:
                pc_attribs:
                pc_augm:
                pc_augm_config:
            '''
            super(PseudoDataset).__init__()
            self.data_path = data_path # train dataset
            self.classes = classes # train classes name
            self.num_point = num_point # 2048
            self.pc_attribs = pc_attribs # xyzrgbXYZ
            self.pc_augm = pc_augm
            self.pc_augm_config = pc_augm_config
            self.bg_data_path = bg_data_path

            train_block_names = []
            all_block_names = []
            for k, v in sorted(class2scans.items()):
                all_block_names.extend(v)
                n_blocks = len(v)
                n_test_blocks = int(n_blocks * 0.1)
                n_train_blocks = n_blocks - n_test_blocks
                train_block_names.extend(v[:n_train_blocks])

            if mode == 'train':
                self.block_names = list(set(all_block_names)) # use all training data.
            elif mode == 'test':
                self.block_names = list(set(all_block_names) - set(train_block_names))
            else:
                raise NotImplementedError('Mode is unknown!')

            self.block_names = sorted(self.block_names)
            print('[Pretrain Dataset] Mode: {0} | Num_blocks: {1}'.format(mode, len(self.block_names)))

        def __len__(self):
            return len(self.block_names)

        def __getitem__(self, index):
            block_name = self.block_names[index]
            # print(block_name)
            ptcloud, label = sample_pointcloud_pseudo(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                                                                self.pc_augm_config, block_name, self.classes, random_sample=True,
                                                                                bg_data_path=self.bg_data_path)

            return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64)), torch.from_numpy(label.astype(np.float32))