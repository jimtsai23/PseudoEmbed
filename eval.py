import os
import random
import numpy as np
import logging
import argparse
import urllib

import torch

import eval_metric as metric
from eval_util import export_pointcloud, convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from label_constants import *
from glob import glob
from tqdm import tqdm
from pathlib import Path


def get_palette(colormap='s3dis13'):
    if colormap == 's3dis13':
        scannet_palette = []
        for _, value in S3DIS_COLOR_MAP_13.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 's3dis6':
        scannet_palette = []
        for _, value in S3DIS_COLOR_MAP_6.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    return palette

def preprocess_xyz(xyz):
    '''make xyz 0 to 1, then center at origin (-0.5,0.5)'''
    xyz = xyz - np.amin(xyz, axis=0)
    xyz = xyz / xyz.max()
    center = np.amax(xyz, axis=0) / 2
    xyz = xyz - center
    return xyz

def load_scene(scene_path):
    scene = np.load(scene_path)
    label = scene[:,6:7]
    rgb = scene[:,3:6]
    xyz_raw = scene[:,0:3]
    xyz = preprocess_xyz(xyz_raw)
    return xyz, rgb, label

def pad_feat3d(feat3d, mask3d):
    feat3d_new = torch.zeros((mask3d.shape[0], feat3d.shape[1]), dtype=feat3d.dtype)
    feat3d_new[mask3d] = feat3d
    return feat3d_new

def evaluate(args):
    '''Evaluate our OpenScene model.'''
    labelset_name = args.labelset_name
    if 's3dis13' in labelset_name:
        labelset = list(S3DIS_LABELS_13)
        labelset[-1] = 'other' # change 'clutter' to 'other'
        palette = get_palette(colormap='s3dis13')
    elif 's3dis6' in labelset_name:
        labelset = list(S3DIS_LABELS_6)
        palette = get_palette(colormap='s3dis6')
    elif 'scannet200' in labelset_name:
        labelset = list(SCANNET_LABELS_200)
        # palette = get_palette(colormap='scannet21')
    elif 'mix' in labelset_name:
        # labelset = list(MIX)
        labelset = ['girder','counter','pillar','window','cabinet','board']
        # palette = get_palette(colormap='scannet21')
    elif 'scannet21' in labelset_name:
        labelset = list(SCANNET_LABELS_21)
        # palette = get_palette(colormap='scannet21')
    elif 'scannet6' in labelset_name:
        labelset = list(SCANNET_LABELS_6)
        # palette = get_palette(colormap='scannet6')
    text_features = extract_text_feature(labelset, args)

    preds, gts, masks = [], [], []


    if 's3dis' in args.dataset:
        scene_dir = f'datasets/S3DIS/scenes/{args.split}_data'
        feat3d_dir = f'z_openseg/{args.dataset}/{args.exp}/{args.split}'
    elif 'scannet' in args.dataset:
        scene_dir = f'datasets/ScanNet/scenes/{args.split}_data'
        feat3d_dir = f'z_openseg_{args.dataset}/{args.exp}/{args.split}'
    # scene_path_lst = sorted(glob(f'{scene_dir}/*'))
    feat3d_path_lst = sorted(glob(f'{feat3d_dir}/*'))
    for i, feat3d_path in enumerate(tqdm(feat3d_path_lst)):
        processed_feat3d = torch.load(feat3d_path)
        scene_id = feat3d_path.split('/')[-1][:-5]
        xyz, rgb, label = load_scene(f'{scene_dir}/{scene_id}.npy')
        feat3d_masked, mask3d = processed_feat3d['feat'], processed_feat3d['mask_full']
        feat3d = pad_feat3d(feat3d_masked, mask3d)
        predictions = feat3d.cuda()
        pred = predictions @ text_features.t()
        logits_pred = torch.max(pred, 1)[1].detach().cpu()
        preds.append(logits_pred)
        gts.append(label)
        masks.append(mask3d)
        # if i>3:
        #     break
        if False:
            if args.mark_no_feature_to_unknown: # this is about visualization
                logits_pred[~mask3d] = len(labelset)-1
            if args.vis_input:
                input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
                export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), xyz, colors=(input_color+1)/2)

            if args.vis_pred:
                pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
                export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), xyz, colors=pred_label_color)
                visualize_labels(list(np.unique(logits_pred.numpy())),
                            labelset,
                            palette,
                            os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)

            if args.vis_gt:
                # for points not evaluating
                label[label==255] = len(labelset)-1
                gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
                export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), xyz, colors=gt_label_color)
                visualize_labels(list(np.unique(label.cpu().numpy())),
                            labelset,
                            palette,
                            os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

    if args.cluster_filter:
        # pred = torch.cat(preds)
        # cls_pnts = torch.bincount(pred)
        # robust_pnt, robust_ind = torch.sort(cls_pnts, descending=True)
        # for j in range(6):
        #     print(labelset[robust_ind[j]],robust_pnt[j])
        # robust_text_features = text_features[robust_ind[:6]]
        robust_text_features = text_features
        generate_pseudo_mask(args,robust_text_features)
        return        


    if args.eval_iou:
        gt = np.concatenate(gts).squeeze(1).astype(np.int64)
        pred = torch.cat(preds)
        pred_logit = pred
        if args.mark_no_feature_to_unknown:
            mask = torch.cat(masks)
            pred_logit[~mask] = 256
        if args.labelset_name=='s3dis6':
            novel_cls_lst = [7,5,4,3,11,9]
            new_gt = np.zeros_like(gt) -1
            for novel_cls in novel_cls_lst:
                new_gt[gt==novel_cls] = novel_cls_lst.index(novel_cls)
            gt = new_gt[new_gt>-1]
            pred_logit = pred_logit[new_gt>-1]
        if args.labelset_name=='scannet6':
            novel_cls_lst = [9,10,11,13,16,18]
            new_gt = np.zeros_like(gt) -1
            for novel_cls in novel_cls_lst:
                new_gt[gt==novel_cls] = novel_cls_lst.index(novel_cls)
            gt = new_gt[new_gt>-1]
            pred_logit = pred_logit[new_gt>-1]

        current_iou = metric.evaluate(pred_logit.numpy(),
                                    gt,
                                    dataset=labelset_name,
                                    stdout=True)
        

def generate_pseudo_mask(args, text_features=None):
    if text_features is None:
        '''Evaluate our OpenScene model.'''
        labelset_name = args.labelset_name
        if 's3dis13' in labelset_name:
            print('error!!!!')
            labelset = list(S3DIS_LABELS_13)
            labelset[-1] = 'other' # change 'clutter' to 'other'
        elif 's3dis6' in labelset_name:
            labelset = list(S3DIS_LABELS_6)
        elif 'scannet6' in labelset_name:
            labelset = list(SCANNET_LABELS_6)
        text_features = extract_text_feature(labelset, args)

    

    if 's3dis' in args.dataset:
        scene_dir = f'datasets/S3DIS/scenes/{args.split}_data'
        feat3d_dir = f'z_openseg/{args.dataset}/{args.exp}/{args.split}' # /{args.split}'
        out_path = 'datasets/S3DIS/scenes/' + args.out
    elif 'scannet' in args.dataset:
        scene_dir = f'datasets/ScanNet/scenes/{args.split}_data'
        feat3d_dir = f'z_openseg_{args.dataset}/{args.exp}/{args.split}'
        out_path = 'datasets/ScanNet/scenes/' + args.out
    
    Path(out_path).mkdir(parents=True,exist_ok=True)
    scene_path_lst = sorted(glob(f'{scene_dir}/*'))
    for i, scene_path in enumerate(tqdm(scene_path_lst)):
        scene = np.load(scene_path)
        scene_id = scene_path.split('/')[-1][:-4]
        try:
            processed_feat3d = torch.load(f'{feat3d_dir}/{scene_id}_0.pt')
            feat3d_masked, mask3d = processed_feat3d['feat'], processed_feat3d['mask_full']
            feat3d = pad_feat3d(feat3d_masked, mask3d)
            predictions = feat3d.cuda()
            pred = predictions @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].detach().cpu().numpy()
            logits_pred[~mask3d] = -1
        except:
            logits_pred = np.zeros([scene.shape[0]]) - 1
        print(np.unique(logits_pred))
        new_scene = np.concatenate([scene,logits_pred[:,None]],axis=1)
        np.save(f'{out_path}/{scene_id}', new_scene)

def get_parser():
    '''Parse the config file.'''
    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--prompt_eng', default=True)
    parser.add_argument('--eval_iou', default=True)
    parser.add_argument('--mark_no_feature_to_unknown', default=True)
    parser.add_argument('--feature_2d_extractor', type=str, default='openseg')
    parser.add_argument('--labelset_name', type=str, default='mix')
    parser.add_argument('--dataset', type=str, default='s3dis')
    parser.add_argument('--exp', type=str, default='base0')
    parser.add_argument('--out', type=str, default='base0_mix')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--generate_pseudo', default=False)
    parser.add_argument('--cluster_filter', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    if args.generate_pseudo:
        generate_pseudo_mask(args)
    else:
        evaluate(args)
