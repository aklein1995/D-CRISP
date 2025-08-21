import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.segmentation import slic
from shapely.geometry import Polygon


def calculate_iou(polygon1, polygon2):
    """
    Calculate Intersection over Union (IoU) of two polygons.
    """
    
    # print('polygon1:',polygon1)
    # print('polygon2:',polygon2)
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)

    if not poly1.is_valid or not poly2.is_valid:
        print('not valid')
        return 0.0

    # Calculate intersection area 
    inter_area = poly1.intersection(poly2).area
    # Calculate union area
    union_area = poly1.union(poly2).area
    # print('inter_area:',inter_area)
    # print('union area:',union_area)
    
    # Compute the IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

def minimal_subset(arr,k_threshold=0.7):
    """
     Records the smallest value for k that causes the prediction of the model to change
     
     arr (numpy.ndarray): scores given after removing pixels iteratively
    """
    if arr[0]<k_threshold:
        return 0
    
    mask = arr<k_threshold
    if mask.any():
        return np.argmax(mask)/len(arr)
    else:
        return 1

# Adapted from: https://github.com/Binh24399/D-CLOSE/blob/main/evaluation.py
def correspond_box(predictbox, predict_classes, groundtruthboxes, groundtruth_classes, iou_th=0.1):
    '''
    predictbox: list of predicted bounding boxes, each defined by 4 vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    predict_classes: list of predicted class labels corresponding to each predicted bounding box
    groundtruthboxes: list of ground-truth bounding boxes, each defined by 4 vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    groundtruth_classes: list of ground-truth class labels corresponding to each ground-truth bounding box
    
    Return: The ground-truth box matches the prediction box and the corresponding index of the prediction box.
    '''
    gt_boxs = []
    det = np.zeros(len(groundtruthboxes)) # there could be more BB-predictions than GT-BB --> assure that a GT-BB are used only once
    idx_predictbox = []

    for d in range(len(predictbox)):
        iouMax = 0
        index = -1
        for i in range(len(groundtruthboxes)):
            if predict_classes[d] != groundtruth_classes[i]:
                continue
            iou = calculate_iou(predictbox[d], groundtruthboxes[i])
            if iou > iouMax:
                iouMax = iou
                index = i

        if iouMax > iou_th:
            if det[index] == 0:
                det[index] = 1
                gt_boxs.append(groundtruthboxes[index])
                idx_predictbox.append(d)
    
    return np.array(gt_boxs), idx_predictbox
