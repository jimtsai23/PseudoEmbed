
import os
import random
import math
import glob
import numpy as np
import h5py as h5
import transforms3d
from itertools import  combinations

import torch
from torch.utils.data import Dataset
import pickle


def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                      sampled_classes, sampled_class=0, support=False, random_sample=False, 
                      use_all_classes=False, bg_data_path=None, return_ori_data=False):
    '''
    Args:
        data_path:
        num_point:
        pc_attribs:
        pc_augm:
        pc_augm_config:
        scan_name:
        sampled_classes:
        sampled_class:
        support:
        random_sample:
        use_all_classes: in base stage of mpti (pretrain+meta-train), we keep a 'bg' class and set its idx 0. In novel stage and testing stage, there is no reservation of 'bg'!

    Returns:

    '''
    sampled_classes = list(sampled_classes) # pre-train: train classes.  train: 2-way classes name
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

    # # get segment label. not used
    # if data.shape[1] == 8:
    #     segment_label = data[:, 7] # (n,)
    # else:
    #     segment_label = np.zeros(data.shape[0], dtype=data.dtype)
    if bg_data_path is not None:
        pseudo_label = data[:, 7]
    else:
        pseudo_label = np.zeros(data.shape[0], dtype=data.dtype)
    if return_ori_data:
        return ptcloud, groundtruth, pseudo_label, data #segment_label

    return ptcloud, groundtruth, pseudo_label #segment_label


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


class SelfDataset(Dataset):
    def __init__(self, data_path, mode='base',
                        num_point=4096, pc_attribs='xyz',
                        pc_augm=False, pc_augm_config=None):
        
        super().__init__()
        self.data_path = data_path # train dataset
        self.mode = mode
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs # xyzrgbXYZ
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        #
        self.block_names = sorted(os.listdir(os.path.join(data_path,'self')))

    def __len__(self):
        return len(self.block_names)
    
    def prepare_data(self,data):
        xyz = data[:, 0:3]
        rgb = data[:, 3:6]

        xyz_min = np.amin(xyz, axis=0)
        xyz -= xyz_min
        if self.pc_augm:
            xyz = augment_pointcloud(xyz, self.pc_augm_config)
        if 'XYZ' in self.pc_attribs:
            xyz_min = np.amin(xyz, axis=0)
            XYZ = xyz - xyz_min
            xyz_max = np.amax(XYZ, axis=0)
            XYZ = XYZ/xyz_max

        ptcloud = []
        if 'xyz' in self.pc_attribs: ptcloud.append(xyz)
        if 'rgb' in self.pc_attribs: ptcloud.append(rgb/255.)
        if 'XYZ' in self.pc_attribs: ptcloud.append(XYZ)
        ptcloud = np.concatenate(ptcloud, axis=1) # (2048, 9)

        return ptcloud


    def __getitem__(self, index):
        block_name = self.block_names[index]
        block_name = os.path.join(self.data_path, 'self', block_name)
        data = np.load(block_name) # not 'self'
        valid_idx = np.nonzero(data[:,8]==0)[0]
        sampled_point_inds = np.random.choice(valid_idx, self.num_point, replace=(data.shape[0] < self.num_point))
        sample_data = data[sampled_point_inds]
        ptcloud = self.prepare_data(sample_data)
        labels = sample_data[:, 6].astype(np.int)
        #
        pseudo_mask = data[:,8]>0
        if pseudo_mask.sum()==0:
            pseudo_pc = np.zeros_like(ptcloud)
            pseudo_label = np.zeros_like(labels)
        else:
            if pseudo_mask.sum() > self.num_point:
                valid_idx = np.nonzero(data[:,8]>0)[0]
                sampled_point_inds = np.random.choice(valid_idx, self.num_point, replace=False)
                pseudo_data = data[sampled_point_inds]
            else:
                pseudo_data = data[pseudo_mask]
            pseudo_pc = self.prepare_data(pseudo_data)
            pseudo_label = pseudo_data[:,7].astype(np.int)
        

        return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(labels.astype(np.int64)), \
                block_name, sampled_point_inds, \
                torch.from_numpy(pseudo_pc.transpose().astype(np.float32)), torch.from_numpy(pseudo_label.astype(np.int64))


def add_identifier(data_path='datasets/S3DIS/blocks_bs1_s1'):
    import pathlib
    save_path = os.path.join(data_path,'self')
    pathlib.Path(save_path).mkdir(parents=True,exist_ok=True)
    dirs = os.listdir(os.path.join(data_path,'data'))
    for d in dirs:
        if d[-4:]=='.npy':
            print(d)
            data = np.load(os.path.join(data_path,'data',d))
            data_id = np.concatenate([data,np.zeros([data.shape[0],2])],axis=-1)
            np.save(os.path.join(save_path,d),data_id)
    print('done')

if __name__ == '__main__':
    add_identifier()