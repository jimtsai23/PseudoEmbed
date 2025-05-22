'''IoU'''
import numpy as np
from label_constants import *

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
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
    return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 's3dis13' in dataset:
        CLASS_LABELS = S3DIS_LABELS_13
    elif 's3dis6' in dataset:
        CLASS_LABELS = S3DIS_LABELS_6
    elif 'scannet21' in dataset:
        CLASS_LABELS = SCANNET_LABELS_21
    elif 'scannet6' in dataset:
        CLASS_LABELS = SCANNET_LABELS_6
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    print(confusion)
    class_ious = {}
    class_accs = {}
    class_precisions = {}
    mean_iou = 0
    mean_acc = 0
    mean_precision = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        class_precisions[label_name] = class_ious[label_name][1] / (pred_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]
        mean_precision += class_precisions[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    mean_precision /= N_CLASSES
    if stdout:
        print('classes    iou    recall    precision')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f} {2:>5.3f} {3:>5.3f}'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_accs[label_name],
                        class_precisions[label_name]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
        print('Mean Precision', mean_precision)
        if 's3dis13' in dataset:
            novel_cls_lst = [7,5,4,3,11,9]
            novel_cls = sorted(novel_cls_lst)
            base_cls = [x for x in np.arange(13) if x not in novel_cls]
            print('base_cls', base_cls)

    return mean_iou
