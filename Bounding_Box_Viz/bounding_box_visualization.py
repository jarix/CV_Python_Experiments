"""
   Draw ground truth bounding boxes onto images
"""

import os
import glob
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def read_data():
    """
       Helper function to read the Ground Truth bounding boxes from JSON
    """
    with open('../Images/traffic_ground_truth.json') as f:
        g_truths = json.load(f)

    return g_truths


def vizualize_bounding_boxes(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    paths = glob.glob('../Images/Traffic/*')

    # map
    gtdic = {}
    for gt in ground_truth:
        gtdic[gt['filename']] = gt

    # color maps
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}

    f, ax = plt.subplots(4, 5, figsize=(20, 10))
    for i in range(20):
        x = i % 4
        y = i % 5

        filename = os.path.basename(paths[i])
        img = Image.open(paths[i])
        ax[x, y].imshow(img)

        bboxes = gtdic[filename]['boxes']
        classes = gtdic[filename]['classes']
        for cl, bb in zip(classes, bboxes):
            y1, x1, y2, x2 = bb
            rec = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor=colormap[cl])
            ax[x, y].add_patch(rec)

        ax[x, y].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ground_truth = read_data()
    vizualize_bounding_boxes(ground_truth)
