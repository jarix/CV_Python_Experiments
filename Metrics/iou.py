"""
  Calculate iou of 2 bounding boxes
"""

import numpy as np
import json

def read_data():
    """
       Helper function to read the Ground Truth and Prediction bounding boxes
    """
    with open('../Images/traffic_ground_truth.json') as f:
        gtruths = json.load(f)

    with open('../Images/traffic_predictions.json') as f:
        preds = json.load(f)

    return gtruths, preds


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    x0 = np.max([gt_bbox[0], pred_bbox[0]])
    y0 = np.max([gt_bbox[1], pred_bbox[1]])
    x1 = np.min([gt_bbox[2], pred_bbox[2]])
    y1 = np.min([gt_bbox[3], pred_bbox[3]])

    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

    union = gt_area + pred_area - intersection

    iou = intersection / union

    return iou


if __name__ == "__main__": 
    ground_truth, predictions = read_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    print(f"ious:\n{ious}")
