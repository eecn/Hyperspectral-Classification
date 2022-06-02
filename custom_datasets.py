from utils import open_file
import numpy as np
import cv2
CUSTOM_DATASETS_CONFIG = {
         'DFC2018_HSI': {
            'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
            'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
            'download': False,
            'loader': lambda folder: dfc2018_loader(folder)
            }
    }


def dfc2018_loader(folder):
        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:,:,:-2]
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        gt = gt.astype('uint8')
        # The original data img size(601, 2384, 50) gt size(1202, 4768)
        # So you first need to downsample the img data or upsample the gt data
        gt = cv2.resize(gt, dsize=(img.shape[0],img.shape[1]), interpolation=cv2.INTER_NEAREST)
        # img  = cv2.resize(img, dsize=(gt.shape[0],gt.shape[1]), interpolation=cv2.INTER_CUBIC)

        rgb_bands = (47, 31, 15)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
        ignored_labels = [0]
        palette = None
        return img, gt, rgb_bands, ignored_labels, label_values, palette
