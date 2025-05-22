import os
import random
import time
# import cv2
import numpy as np
import logging
import argparse
import random
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data


from torch.utils.tensorboard import SummaryWriter

from model.capl_pseudo import CAPL
from util.util import AverageMeter #, poly_learning_rate, intersectionAndUnionGPU
from dataloaders.loader import Testing_Dataset, ValSupp_Dataset
from dataloaders.loader_pseudo import TrainDataset
from runs.eval import evaluate_metric_GFS
from runs.eval_pseudo import evaluate
from util.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
import ast
from util.logger import init_logger
from tqdm import tqdm
from pathlib import Path

def get_class_wise_iou(mean_iou_list, logger):
    '''
    Args:
        mean_iou_list: [[iou_list seed 10], [iou_list seed 20], ...]
        logger:
    Returns:
    '''
    stack_iou = np.stack(mean_iou_list, axis=0) # (5, num_class)
    stack_iou = np.mean(stack_iou, axis=0) # (num_class, )
    num_class = len(mean_iou_list[0])
    for i in range(num_class):
        if args.dataset=='scannet':
            logger.cprint('class {}, iou over multiple runs: {}'.format(i+1, stack_iou[i]))
        else:
            logger.cprint('class {}, iou over multiple runs: {}'.format(i, stack_iou[i]))

def get_new_proto(val_supp_loader, model, base_num=16, novel_num=5, novel_class_list=None):
    ''' get base and novel protoes. base proto is via eqn.3. novel proto is done by eqn.1
        Args:
            val_supp_loader:
            model:
            base_num:
            novel_num:
            novel_class_list: learning order idx of new classes. [7,8,9,10,11]
            proto_dim: feature dimension of the prototypes
        Returns: proto of all classes (cls, 192)

    '''

    logger.cprint('>>>>>>>>>>>>>>>> Start New Proto Generation >>>>>>>>>>>>>>>>')

    model.eval()
    new_proto_num_epoch = 1  # 1
    with torch.no_grad():
        proto_dim = model.main_proto.shape[1]
        total_classes = base_num + novel_num
        gened_proto_bed = torch.zeros(total_classes, proto_dim).cuda()  # empty proto. (21, 512).
        for epoch in range(new_proto_num_epoch):
            new_cls_feat_dict = {cls: [] for cls in novel_class_list}
            for i, (input, target, cls_id) in enumerate(val_supp_loader): # cls_id is idx!!
                input = input.cuda()
                target = target.cuda()
                # get feature
                cls_feat = model.Get_Fg_Feat(x=input, y=target)  # collect class feature. (n, d)
                cls_feat = torch.mean(cls_feat, dim=0, keepdim=True) # add this (1, d)
                new_cls_feat_dict[cls_id[0].item()].append(cls_feat)

            # initialize final proto
            gened_proto = torch.zeros_like(model.main_proto, requires_grad=False) # (cls, d)
            assert gened_proto.device == model.main_proto.device
            # copy base proto
            gened_proto[:base_num, :] = model.main_proto[:base_num, :].detach().clone()
            # get novel proto
            for cls in novel_class_list:
                gened_proto[cls,:] = torch.mean(torch.cat(new_cls_feat_dict[cls], dim=0), dim=0, keepdim=False) # (d,)

            gened_proto = F.normalize(gened_proto, p=2, dim=1)  # l2 norm.
            gened_proto_bed = gened_proto_bed + gened_proto
        gened_proto = gened_proto_bed / new_proto_num_epoch

    return gened_proto  # (cls, 192)

def check_proto_dist(novel_proto, proto_lst):
    novel_proto = novel_proto / novel_proto.norm(dim=1,keepdim=True)
    torch.set_printoptions(precision=1, linewidth=150, sci_mode=False)
    # torch.set_printoptions(precision=3, linewidth=100, sci_mode=False)
    novel_proto = proto_lst[0][-6:]
    for proto in proto_lst:
        print(novel_proto @ proto.T)
        print('\n')
    # breakpoint()

def main(argss, basis_path):
    global args
    args = argss
    global logger, writer
    logger = init_logger(args.save_path, args)

    if args.ignore_bg:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ------------------- define validation dataloader --------------------.
    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.testing_data_path)
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.testing_data_path)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    train_class_names = sorted(DATASET.train_classes)  # they are sorted by the class name order. # [0, 1, 2, 6, 8, 10, 12]
    test_class_names = sorted(DATASET.test_classes)  # sorted by class_name order. # [3, 4, 5, 7, 9, 11]

    all_learning_order = sorted(DATASET.train_classes)
    all_learning_order.extend(test_class_names)  # learning order of all classes # [0, 1, 2, 6, 8, 10, 12, 3, 4, 5, 7, 9, 11]
    all_class_names = sorted(all_learning_order)  # all classes sorted by class name order. # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # get learning order idx of testing classes
    test_learning_order_idx = []
    for i in test_class_names:
        test_learning_order_idx.append(all_learning_order.index(i)) # [7, 8, 9, 10, 11, 12]

    print('testing classes : {}'.format(all_class_names))  # make sure CLASSES are in order before sending them into dataset.

    test_CLASS2SCANS = {c: DATASET.class2scans[c] for c in all_class_names}  # only use the train classes in class2scan
    # print('test class2scans')
    # for c in test_CLASS2SCANS.keys():
    #     print(c,len(test_CLASS2SCANS[c]))

    VALID_DATASET = Testing_Dataset(args.testing_data_path, all_class_names, all_learning_order, test_CLASS2SCANS,
                                    mode='test',
                                    num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                    pc_augm=False,
                                    visualize=args.visualize)

    val_bs = 1 if args.visualize else args.batch_size
    val_loader = torch.utils.data.DataLoader(VALID_DATASET, batch_size=val_bs, num_workers=args.n_workers, shuffle=False,
                                             drop_last=False)
    
    model = CAPL(classes=len(all_class_names), criterion=criterion, args=args, base_cls=train_class_names, novel_cls=test_class_names, use_grad=args.use_grad)
    model = model.cuda()
    if not args.openset:
        # get novel class dataset: support set. label is binary only.
        seed_list = [10, 20, 30, 40, 50]
        val_supp_loader_list = []
        for seed in seed_list:
            val_supp_data = ValSupp_Dataset(data_path=args.data_path, dataset_name=args.dataset, cvfold=args.cvfold,
                                            k_shot=args.k_shot, mode='test',
                                            num_point=args.pc_npts, pc_attribs=args.pc_attribs, pc_augm=False,
                                            pc_augm_config=None, seed=seed, learning_order=all_learning_order)

            val_supp_loader = torch.utils.data.DataLoader(val_supp_data, batch_size=1, shuffle=False,
                                                        num_workers=args.n_workers, drop_last=False)
            val_supp_loader_list.append(val_supp_loader)

        if args.only_evaluate:
            logger.cprint('--------- loading weight for evaluation -----------')
            model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')
            if args.check_proto:
                proto_lst = []
                for val_supp_loader in val_supp_loader_list:
                    gened_proto = get_new_proto(val_supp_loader, model, novel_num=len(test_class_names), base_num=len(train_class_names), \
                                                novel_class_list=test_learning_order_idx)
                    proto_lst.append(gened_proto)
                novel_proto = model.main_proto.clone().detach()[-6:]
                check_proto_dist(novel_proto, proto_lst)
                return

            mean_mean_mIoU = 0
            mean_base_mIoU = 0
            mean_novel_mIoU = 0
            mean_hm_mIoU = 0
            mean_iou_list = []
            for val_supp_loader in val_supp_loader_list:
            
                gened_proto = get_new_proto(val_supp_loader, model, novel_num=len(test_class_names), base_num=len(train_class_names), \
                                            novel_class_list=test_learning_order_idx)
                mean_iou, base_iou, novel_iou, hm_iou, iou_list = validate(val_loader, model, novel_num=len(test_class_names),
                                                        base_num=len(train_class_names),
                                                        gened_proto=gened_proto.clone(),
                                                        all_classes=all_class_names, novel_classes=test_class_names,
                                                        all_learning_order=all_learning_order)

                mean_mean_mIoU += mean_iou
                mean_base_mIoU += base_iou
                mean_novel_mIoU += novel_iou
                mean_hm_mIoU += hm_iou
                mean_iou_list.append(iou_list)
            mIoU_val = mean_mean_mIoU / len(val_supp_loader_list)
            base_mIoU = mean_base_mIoU / len(val_supp_loader_list)
            novel_mIoU = mean_novel_mIoU / len(val_supp_loader_list)
            hm_mIoU = mean_hm_mIoU / len(val_supp_loader_list)
            logger.cprint(f'Eval result: BASE: {base_mIoU:.4f}, NOVEL: {novel_mIoU:.4f}, avg: {mIoU_val:.4f}, hm_mIoU: {hm_mIoU:.4f}')
            # logger.cprint('Eval result: Final mIoU: {}, BASE: {}, NOVEL: {}, hm_mIoU: {}'.format(mIoU_val, base_mIoU, novel_mIoU, hm_mIoU))
            # print class-wise mean iou:
            get_class_wise_iou(mean_iou_list, logger)
            exit(0)

    else:
        num_point = 2048
        model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')
        DATASET = S3DISDataset(args.cvfold, args.data_path, args.bg_data_path)  # class2scan is defferent from testing dataset!!!
        pseudo_CLASS2SCANS = {c: DATASET.pseudo_class2scans[c] for c in range(6)}
        psuedo_prototypes = []
        for sampled_class, name_lst in pseudo_CLASS2SCANS.items():
            # psuedo_prototypes.append(torch.zeros([1,192]).cuda())
            # continue
            feat_lst = []
            for k, block_name in enumerate(tqdm(name_lst)):
                try:
                    data = np.load(os.path.join(args.data_path, args.bg_data_path, '%s.npy' %block_name))
                    valid_point_inds = np.nonzero(data[:,7] == sampled_class)[0]  # indices of points belonging to the sampled class
                    if len(valid_point_inds) > num_point:
                        valid_point_inds = np.random.choice(valid_point_inds, num_point, replace=False)
                    if len(valid_point_inds) < 200:
                        continue
                    data = data[valid_point_inds]
                    xyz = data[:, 0:3]
                    rgb = data[:, 3:6]
                    #
                    xyz_min = np.amin(xyz, axis=0)
                    xyz -= xyz_min
                    #
                    xyz_min = np.amin(xyz, axis=0)
                    XYZ = xyz - xyz_min
                    xyz_max = np.amax(XYZ, axis=0)
                    XYZ = XYZ/xyz_max
                    #
                    ptcloud = np.concatenate([xyz, rgb/255., XYZ], axis=1) # (2048, 9)
                    input = torch.from_numpy(ptcloud.transpose().astype(np.float32))[None].cuda()
                    with torch.no_grad():
                        feat = model.pseudo_proto_feat(input) # (n,d)
                    del input
                    feat_lst.append(feat.cpu())
                    # if k>2000: break
                    if k>1000: break
                except:
                    continue

            feat_lst = torch.cat(feat_lst,dim=0)
            proto = torch.mean(feat_lst, dim=0, keepdim=True) # (1, d)
            psuedo_prototypes.append(proto.cuda())

        base_num=len(train_class_names)
        gened_proto = torch.zeros_like(model.main_proto, requires_grad=False) # (cls, d)
        gened_proto[:base_num, :] = model.main_proto[:base_num, :].detach().clone()
        for i, proto in enumerate(psuedo_prototypes):
            gened_proto[base_num+i] = proto

        openset(val_loader, model, novel_num=len(test_class_names),
                base_num=len(train_class_names),
                gened_proto=gened_proto.clone(),
                all_classes=all_class_names, novel_classes=test_class_names,
                all_learning_order=all_learning_order)
        return




    # --------------------------- TRAINING ----------------------------------- Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }

    # redefine training dataset !!! the data path is defferent
    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.data_path, args.bg_data_path)  # class2scan is defferent from testing dataset!!!
        # pseudo_order = [7,5,4,3,11,9]
        # novel = [3, 4, 5, 7, 9, 11]
        pseudo2novel = [3,2,1,0,5,4]
        train_CLASS2SCANS = {c: DATASET.class2scans[c] for c in all_class_names}
        pseudo_CLASS2SCANS = {c: DATASET.pseudo_class2scans[c] for c in range(len(pseudo2novel))}
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.data_path, args.bg_data_path)
        # pseudo_order = [9,10,11,13,16,18]
        # novel = [9,10,11,13,16,18]
        pseudo2novel = [0,1,2,3,4,5]
        train_CLASS2SCANS = {c: DATASET.class2scans[c] for c in train_class_names}
        pseudo_CLASS2SCANS = {c: DATASET.pseudo_class2scans[c] for c in range(len(pseudo2novel))}
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    if args.up_bnd: print('use up_bnd!')
    train_data = TrainDataset(args.data_path, train_class_names, test_class_names, train_CLASS2SCANS, pseudo_CLASS2SCANS, num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                   pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG, bg_data_path=args.bg_data_path, up_bnd=args.up_bnd, pseudo2novel=pseudo2novel)
    logger.cprint('=== Pre-train Dataset (classes: {0}) | Train: {1} blocks | Valid: {2} blocks ==='.format(train_class_names, len(train_data), 0))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, drop_last=True)

    # define optimizer
    optimizer = torch.optim.Adam(
                    [
                     {'params': model.encoder.parameters(), 'lr':0.1*args.base_lr},
                     {'params': model.base_learner.parameters()},
                     {'params': model.att_learner.parameters()},
                     {'params': model.main_proto},
                     {'params': model.bg_proto},
                        # {'params': model.fusion.parameters()},
                     ],
                        lr=args.base_lr)
    if args.resume:
        model, optimizer = load_model_checkpoint(model, args.model_checkpoint_path, optimizer=optimizer, mode='train')

    # set learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                  gamma=args.gamma)

    # load pretrain weight of feature extractor:
    if args.use_pretrain_weight and not args.resume:
        logger.cprint('----- loading pretrain weight of feature extractor --------')
        model = load_pretrain_checkpoint(model, args.pretrain_checkpoint_path)
        # model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')


    writer = SummaryWriter(args.save_path)
    max_iou = 0.
    filename = 'capl.pth'
    max_hm =0.
    hm_filename = 'hm.pth'
    for epoch in range(args.start_epoch, args.epochs):
       
        train(train_loader, model, optimizer, epoch, lr_scheduler)

        if args.evaluate and (epoch+1) % 5 == 0:
            mean_mean_mIoU = 0
            mean_base_mIoU = 0
            mean_novel_mIoU = 0
            mean_hm_mIoU = 0
            val_supp_loader_list = val_supp_loader_list[0:1]
            for val_supp_loader in val_supp_loader_list:
                gened_proto = get_new_proto(val_supp_loader, model, novel_num=len(test_class_names), base_num=len(train_class_names),
                                            novel_class_list=test_learning_order_idx)  # get base + novel protoes. skip eqn.3. because we don't have base annotation is novel stage! (13, 192)
                mean_iou, base_iou, novel_iou, hm_iou, _ = validate(val_loader, model,
                                                         novel_num=len(test_class_names),
                                                         base_num=len(train_class_names),
                                                         gened_proto=gened_proto.clone(),
                                                         all_classes=all_class_names, novel_classes=test_class_names,
                                                         all_learning_order=all_learning_order)
                mean_mean_mIoU += mean_iou
                mean_base_mIoU += base_iou
                mean_novel_mIoU += novel_iou
                mean_hm_mIoU += hm_iou
            mIoU_val = mean_mean_mIoU / len(val_supp_loader_list)
            base_mIoU = mean_base_mIoU / len(val_supp_loader_list)
            novel_mIoU = mean_novel_mIoU / len(val_supp_loader_list)
            hm_mIoU = mean_hm_mIoU / len(val_supp_loader_list)
            logger.cprint(f'Epoch {epoch}: BASE: {base_mIoU:.4f}, NOVEL: {novel_mIoU:.4f}, avg: {mIoU_val:.4f}, hm_mIoU: {hm_mIoU:.4f}')
            # logger.cprint('Epoch: {}, Final mIoU: {}, BASE: {}, NOVEL: {}, hm: {}'.format(epoch, mIoU_val, base_mIoU, novel_mIoU, hm_mIoU))

            # writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('Val/mIoU_val', mIoU_val, epoch)
            writer.add_scalar('Val/base_mIoU', base_mIoU, epoch)
            writer.add_scalar('Val/novel_mIoU', novel_mIoU, epoch)
            writer.add_scalar('Val/hm_mIoU', hm_mIoU, epoch)

            if mIoU_val > max_iou:
                max_iou = mIoU_val # mean iou.
                if os.path.exists(filename):
                    os.remove(filename)
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+ f'{max_iou:.4f}'+ \
                            '_Base_'+f'{base_mIoU:.4f}'+'_Novel_'+f'{novel_mIoU:.4f}'+'_hm_'+f'{hm_mIoU:.4f}'+'.pth'
                # filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+ str(max_iou)+'_Base_'+str(base_mIoU)+'_Novel_'+str(novel_mIoU)+'_hm_'+str(hm_mIoU)+'.pth'
                logger.cprint('Saving best checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'max_iou': max_iou}, filename)

            # hm evaluation
            if hm_mIoU > max_hm:
                max_hm = hm_mIoU # mean iou.
                if os.path.exists(hm_filename):
                    os.remove(hm_filename)
                hm_filename = args.save_path + '/train_hm_epoch_' + str(epoch) + '_'+ f'{max_iou:.4f}'+ \
                            '_Base_'+f'{base_mIoU:.4f}'+'_Novel_'+f'{novel_mIoU:.4f}'+'_hm_'+f'{hm_mIoU:.4f}'+'.pth'
                # hm_filename = args.save_path + '/train_hm_epoch_' + str(epoch) + '_'+ str(max_iou)+'_Base_'+str(base_mIoU)+'_Novel_'+str(novel_mIoU)+'_hm_'+str(hm_mIoU)+'.pth'
                logger.cprint('Saving best checkpoint to: ' + hm_filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'max_iou': max_hm}, hm_filename)


def train(train_loader, model, optimizer, epoch, lr_scheduler):
    '''
    Args:
        train_loader:
        model:
        optimizer:
        epoch: current epoch
        lr_scheduler:
        base_class_coding: (n_base, k)

    Returns:

    '''
    torch.cuda.empty_cache()

    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        # print(i, target.unique())
        # continue
        current_iter = epoch * len(train_loader) + i + 1

        input = input.cuda() # (b, d, 2048)
        target = target.cuda() # (b, 2048)

        output, loss_ce = model(x=input, y=target) # output: pred_label. (b, 2048)

        loss = loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.eq(output, target).sum().item()  # including background
        accuracy = correct / (input.shape[0] * input.shape[-1])


        accuracy_meter.update(accuracy)
        loss_meter.update(loss.item(), 1)
        loss_ce_meter.update(loss_ce.item(), 1)


        if (i + 1) % args.print_freq == 0:
            logger.cprint('Epoch: [{}/{}][{}/{}] '                       
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Loss_ce {loss_ce_meter.val:.4f} ({loss_ce_meter.avg:.4f})'
                        'Accuracy {accuracy:.4f} ({accuracy_meter.avg:.4f}).'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          loss_meter=loss_meter, loss_ce_meter=loss_ce_meter,
                                                          accuracy=accuracy, accuracy_meter=accuracy_meter))
        # if i>20:
        #     break

    lr_scheduler.step()

    acc = accuracy_meter.avg
    logger.cprint('Train result at epoch [{}/{}]: acc {:.4f}.'.format(epoch+1, args.epochs, acc))

    # tensorboard
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/loss_ce', loss_ce_meter.avg, epoch)
    writer.add_scalar('Train/accuracy', acc, epoch)


def validate(val_loader, model, novel_num, base_num, gened_proto, all_classes, novel_classes, all_learning_order):
    '''
    Args:
        val_supp_loader:
        val_loader:
        model:
        criterion:
        novel_num:
        base_num:
        gened_proto:
        all_classes: [0,1,2,3,4,5,7,8,9,....]
        novel_classes: novel class names
        all_learning_order: learning order of all classes
        basis: basis (n, d). gpu tensor. no grad.
    Returns:

    '''
    torch.cuda.empty_cache() 

    if len(all_learning_order) > 13:
        use_scannet = True
        print('use scannet, skip class 0!')
    else:
        use_scannet = False

    logger.cprint('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # loss_meter = AverageMeter()

    model.eval()
    # end = time.time()

    pred_labels_list = []
    gt_labels_list = []
    pred_labels_list_geo2sem = []
    xyz_lst = []

    with torch.no_grad():
        time_a = time.time()
        for i, batch_data in enumerate(tqdm(val_loader)):
            if args.visualize:
                input, target, segment_label, xyz_sample, block_name = batch_data
                xyz_sample = xyz_sample[0].numpy()
                block_name = block_name[0]
            else:
                input, target, segment_label = batch_data
            gt_labels_list.append(target.clone().view(-1).numpy())

            input = input.cuda()
            target = target.cuda() # (1, 2048)
            output = model(x=input, y=target, eval_model=True, gened_proto=gened_proto) # logits (b, cls, n)
            query_pred = torch.argmax(output, dim=1, keepdim=False) # (b, n)

            pred_labels_list.append(query_pred.view(-1).cpu().numpy())
            if args.visualize:
                pred = pred_labels_list[-1]
                rgb = np.zeros_like(xyz_sample)
                scene = np.concatenate([xyz_sample,rgb,pred[:,None]],axis=1)
                np.save(f'{args.vis_path}/{block_name}', scene)

            # if i>20:
            #     break

        time_b = time.time()
        # get iou
        pred_ids = np.concatenate(pred_labels_list)
        gt_ids = np.concatenate(gt_labels_list)
        mean_iou, base_iou, novel_iou, hm_iou, iou_list = evaluate(logger, pred_ids, gt_ids, all_classes, novel_classes, all_learning_order, scannet=use_scannet)
        if False:
            mean_iou, base_iou, novel_iou, hm_iou, iou_list = evaluate_metric_GFS(logger, pred_labels_list, gt_labels_list,
                                                                all_classes, novel_classes, all_learning_order, scannet=use_scannet) # need to change the class order.
        time_c = time.time()
        # print('model run time:', time_b - time_a)
        # print('score time:', time_c - time_b)
    return mean_iou, base_iou, novel_iou, hm_iou, iou_list


def openset(val_loader, model, novel_num, base_num, gened_proto, all_classes, novel_classes, all_learning_order):
    
    torch.cuda.empty_cache() 

    logger.cprint('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')


    model.eval()
    # end = time.time()

    gt_labels_list = []
    score_lst = []

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(val_loader)):
            
            input, target, segment_label = batch_data
            gt_labels_list.append(target.clone().view(-1).numpy())

            input = input.cuda()
            target = target.cuda() # (1, 2048)
            output = model(x=input, y=target, eval_model=True, gened_proto=gened_proto) # logits (b, cls, n)
            output = F.softmax(output,dim=1)
            score = output[:,-novel_num:].max(dim=1)[0]
            score_lst.append(score.view(-1).cpu().numpy())
            
            # if i>20:
            #     break

        score = np.concatenate(score_lst)
        label = np.concatenate(gt_labels_list)
        label[label<7] = 0
        label[label>=7] = 1

        # breakpoint()
        from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
        precision, recall, _ = precision_recall_curve(label, score)
        aupr_score = auc(recall, precision)
        print('AUPR is: ', aupr_score)

        fpr, tpr, _ = roc_curve(label, score)
        auroc_score_1 = auc(fpr, tpr)
        # auroc_score_2 = roc_auc_score(label, score)
        print('AUROC is: ', auroc_score_1)

        print('FPR95 is: ', fpr[tpr > 0.95][0])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')

    parser.add_argument('--train_gpu', default=[0]) # doesn't use
    parser.add_argument('--batch_size_val', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manual_seed', type=int, default=321)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='log_s3dis/S0_K5/debug')
    parser.add_argument('--start_val_epoch', type=int, default=25)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--ngpus_per_node', type=int, default=1)
    # data
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str,
                        default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS_Area6AsTest_SP/blocks_bs1.0_s1.0',
                        help='Directory to the source data')
    parser.add_argument('--testing_data_path', type=str,
                        default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS_Area6AsTest_SP/blocks_bs1.0_s1.0_test')
    parser.add_argument('--total_classes', type=int, default=13, help='number of classes to be evaluate in the gfs')
    # model weight
    parser.add_argument('--use_pretrain_weight', action='store_true',
                        help='whether use pretrain weight of the feature extractor')
    parser.add_argument('--pretrain_checkpoint_path', type=str,
                        default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/mpti/log_s3dis/log_pretrain_s3dis_S0_LongTail/',
                        help='Path to the checkpoint of pre model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default='log_s3dis/S0_K5/train_epoch_35_0.3247954127747135_Base_0.4056141974051477_Novel_0.23050683070587352.pth',
                        help='Path to the checkpoint of model for resuming')
    # optimization
    parser.add_argument('--batch_size', type=int, default=16, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=100, help='number of iterations/epochs to train')
    parser.add_argument('--step_size', type=int, default=50, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    # few-shot episode setting
    parser.add_argument('--k_shot', type=int, default=5, help='Number of samples/shots for each class: 1|5')
    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')
    # feature extraction network configuration
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
    parser.add_argument('--edgeconv_widths', default='[[64,64], [64, 64], [64, 64]]', help='DGCNN Edgeconv widths')
    parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]',
                        help='DGCNN MLP (following stacked Edgeconv) widths')
    parser.add_argument('--base_widths', default='[128, 64]', help='BaseLearner widths')  # didn't use in pre-train
    parser.add_argument('--output_dim', type=int, default=64,
                        help='The dimension of the final output of attention learner or linear mapper')  # didn't use in pre-train
    parser.add_argument('--use_attention', action='store_false',
                        help='it incorporate attention learner')  # set to True by default
    # GFS
    parser.add_argument('--seed', default=321, type=int, help='seed')
    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--basis_path', type=str, default='log_s3dis/S0_K5/GlobalKmeans_EdgeConv123_cnt=100_energy=095_SVDReconstruct.pkl', help='path of basis')
    parser.add_argument('--base_class_gp_coding_path', type=str, default='log_s3dis/S0_K5/BaseClass_gp.pkl', help='path of base_class_gp_coding_path')
    parser.add_argument('--energy', type=float, default=0.9, help='frequency limit in alg.1. must <= 1!!')
    parser.add_argument('--eval_weight', type=float, default=1., help='beta weight for re-weighting. validation=1., testing > 1.')
    
    # eccv
    parser.add_argument('--bg_data_path', default=None, type=str)
    parser.add_argument('--up_bnd', action='store_true')
    parser.add_argument('--use_grad', action='store_true', default=False)
    parser.add_argument('--check_proto', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--ignore_bg', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--openset', action='store_true', default=False)

    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert args.energy <= 1
    if args.visualize:
        args.vis_path = f'{args.testing_data_path}/{args.bg_data_path}'
        Path(args.vis_path).mkdir(parents=True,exist_ok=True)

    main(args, basis_path=args.basis_path)
