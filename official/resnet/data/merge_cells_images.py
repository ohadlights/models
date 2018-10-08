import os

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def main():
    data = pd.read_csv(r'X:\cells\train.csv')

    ids = data['Id'].tolist()

    inout_dir = r'X:\cells\train'
    output_dir = r'X:\cells\train_merged'

    for img_id in tqdm(ids):
        img_path = os.path.join(inout_dir, img_id)
        red_img = (img_path + "_red.png")
        yellow_img = (img_path + "_yellow.png")
        blue_img = (img_path + "_blue.png")
        green_img = (img_path + "_green.png")

        image = np.zeros((512, 512, 4), np.uint8)
        image[:, :, 0] = cv2.imread(red_img, cv2.IMREAD_UNCHANGED)
        image[:, :, 1] = cv2.imread(yellow_img, cv2.IMREAD_UNCHANGED)
        image[:, :, 2] = cv2.imread(blue_img, cv2.IMREAD_UNCHANGED)
        image[:, :, 3] = cv2.imread(green_img, cv2.IMREAD_UNCHANGED)

        out_img = os.path.join(output_dir, img_id + "_rgba.png")
        cv2.imwrite(out_img, image)


if __name__ == '__main__':
    main()