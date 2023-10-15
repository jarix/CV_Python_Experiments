"""
  Calculate precision and recall
"""
import numpy as np
import json

from iou import calculate_ious


def read_data():
    """
       Helper function to read the Ground Truth and Prediction bounding boxes
    """
    with open('../Images/traffic_ground_truth.json') as f:
        gtruths = json.load(f)

    with open('../Images/traffic_predictions.json') as f:
        preds = json.load(f)

    return gtruths, preds


def precision_recall(ious, gt_classes, pred_classes):
    """
    calculate precision and recall
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    returns:
    - precision [float]
    - recall [float]
    """
    xs, ys = np.where(ious > 0.5)

    # true positive and false positive
    tp = 0
    fp = 0

    for x, y in zip(xs, ys):
        if gt_classes[x] == pred_classes[y]:
            tp += 1
        else:
            fp += 1

    matched_gt = len(np.unique(xs))
    fn = len(gt_classes) - matched_gt

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


if __name__ == "__main__":
    ground_truth, predictions = read_data()

    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]

    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]

    ious = calculate_ious(gt_bboxes, pred_boxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)

    print(f"Precision: {precision}\nRecall: {recall}")
