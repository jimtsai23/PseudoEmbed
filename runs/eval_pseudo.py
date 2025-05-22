'''IoU'''
import numpy as np
from zlabel_constants import *

# UNKNOWN_ID = 255
# NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    # idxs = gt_ids != UNKNOWN_ID
    # if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
    #     pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
    #     confusion = np.bincount(
    #         pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
    #         minlength=(num_classes+1)**2).reshape((
    #         num_classes+1, num_classes+1)).astype(np.ulonglong)
    #     return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids * num_classes + gt_ids,
        # pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom #, tp, denom


def evaluate(logger, pred_ids, gt_ids, all_classes, novel_classes, all_learning_order, scannet=False):

    N_CLASSES = len(all_classes)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    # np.set_printoptions(linewidth=200)
    # print(confusion)
    class_ious = {}
    class_accs = {}
    class_precisions = {}
    mean_iou = 0
    mean_acc = 0
    mean_precision = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = i
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        # class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        # class_precisions[label_name] = class_ious[label_name][1] / (pred_ids==i).sum()
        # count+=1

    order_iou = {} # following base then novel order
    for i in range(N_CLASSES):
        order_iou[all_learning_order[i]] = class_ious[i]

    iou_list = []
    base_iou_list = []
    novel_iou_list = []
    for i in range(N_CLASSES):
        iou = order_iou[i]
        logger.cprint('----- [class %d]  IoU: %f -----' % (i, iou))
        if scannet and (i==0):
            continue
        if i in novel_classes:
            novel_iou_list.append(iou)
        else:
            base_iou_list.append(iou)
        iou_list.append(iou)

    # get base iou
    base_iou = np.array(base_iou_list).mean()
    logger.cprint('base-iou: {}'.format(base_iou))
    # get novel iou
    novel_iou = np.array(novel_iou_list).mean()
    logger.cprint('novel-iou: {}'.format(novel_iou))
    # get avg iou
    mean_iou = np.array(iou_list).mean()
    logger.cprint('mean-iou: {}'.format(mean_iou))
    # get hm
    hm = 2 * base_iou * novel_iou / (base_iou + novel_iou)
    logger.cprint('hm-iou: {}'.format(hm))

    return mean_iou, base_iou, novel_iou, hm, np.array(iou_list)