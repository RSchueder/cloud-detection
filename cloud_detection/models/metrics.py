import numpy as np

from cloud_detection.utils import get_mask_array


def get_iou(mask, prediction, classes, labels=None):
    """
    Calculates IOU for mulitple classes for a single sample.

    Args:
        mask (array): The target mask.
        prediction (array): The prediction array.
        classes (list(int)): A list of classes to calculate IoU for.
        labels (list(str)): A list of class labels.

    Returns:

    """
    iou_dict = dict()
    if labels is None:
        labels = classes
    for pclass, label in zip(classes, labels):
        mask_flat = mask.flatten()
        r_flat = prediction.flatten()

        intersection = np.sum(np.logical_and(mask_flat == pclass, r_flat == pclass))
        union = np.sum(np.logical_or(mask_flat == pclass, r_flat == pclass))

        if union != 0:
            iou = intersection / union
            iou_dict[label] = iou
        else:
            iou_dict[label] = np.nan

    return iou_dict


def get_mean_iou(mask_paths, predicitions, classes, labels=None):
    """
    Calculates mean IOU per class on a set of predictions.
    Args:
        mask_paths (list): List of mask file paths.
        predicitions (array): An array of predictions (n_samples, H, W, NUM_CLASSES).
        classes (list(int)): A list of classes to calculate IOU for.
        labels (list(str)): A list of class labels.

    Returns:

    """
    iou_dict = dict()
    if labels is None:
        labels = classes
    for pclass, label in zip(classes, labels):
        iou_dict[label] = list()
        for idx in range(len(mask_paths)):
            mask = get_mask_array(mask_paths[idx])
            prediction = get_prediction(predicitions[idx])
            iou = get_iou(mask, prediction, [pclass])
            iou_val = iou[pclass]
            iou_dict[label].append(iou_val)

        iou_dict[label] = np.array(iou_dict[label])
        iou_dict[label] = iou_dict[label][~np.isnan(iou_dict[label])]

    return iou_dict


def get_prediction(prediction):
    """
    Converts probability array to highest probability class based on index.

    Args:
        prediction (array): array of (H, W, NUM_CLASSES)

    Returns:
        array of int with size of (H,W)
    """
    mask = np.argmax(prediction, axis=-1)

    return mask
