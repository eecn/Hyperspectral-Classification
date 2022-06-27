from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "WHUHiLongKou": {
        "img": "WHU_Hi_LongKou.mat",
        "gt": "WHU_Hi_LongKou_gt.mat",
        "download": False,
        "loader": lambda folder: WHULongKou_loader(folder),
    }
}


def WHULongKou_loader(folder):
    img = open_file(folder + "WHU_Hi_LongKou.mat")['WHU_Hi_LongKou']
    gt = open_file(folder + "WHU_Hi_LongKou_gt.mat")['WHU_Hi_LongKou_gt']

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unknown",
        "Corn",
        "Cotton",
        "Sesame",
        "Broad-leaf soybean",
        "Narrow-leaf soybean",
        "Rices",
        "Water",
        "Roads and houses",
        "Mixed weed",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
