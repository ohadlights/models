import os
import json
from shutil import copy

import cv2
import numpy as np
from tqdm import tqdm


resolution = [720, 1280, 3]


def read_bin(fname):
    with open(fname, 'rb') as f:
        buffer = (f.read(resolution[0] * resolution[1] * resolution[2]))
    buffer = np.asarray(list(buffer))
    image = buffer.reshape(resolution).astype('uint8')
    return image


source_dir = r'Z:\PalletWallmart\Original'
out_dir = r'D:\temp\Walmart\PNG'


collected_data = []

for root, dirs, files in os.walk(source_dir):
    json_files = list(filter(lambda f: f.endswith('_v2.json'), files))
    if len(json_files) > 0:
        os.makedirs(root.replace(source_dir, out_dir), exist_ok=True)
    for json_file in json_files:
        collected_data += [(root, json_file)]

with open(r'Z:\PalletWallmart\Train\Lists\train_list.txt', 'w') as f:
    for root, json_file in tqdm(collected_data):
        try:
            json_path = os.path.join(root, json_file)
            dest_json_path = json_path.replace(source_dir, out_dir)
            # if os.path.exists(dest_json_path):
            #     continue
            copy(json_path, dest_json_path)

            bin_path = json_path.replace('_v2.json','.bin')
            image = read_bin(bin_path)

            png_path = bin_path.replace('.bin', '.png').replace(source_dir, out_dir)
            cv2.imwrite(png_path, image)

            with open(json_path, 'r') as json_read_file:
                annotations = json.load(json_read_file)

            f.write('{}'.format(png_path))
            for box in annotations:
                cls = box[4]
                box = [
                    box[0],
                    box[1],
                    box[2] - box[0],
                    box[3] - box[1]
                ]
                f.write(' {},{}'.format(','.join([str(a) for a in box]), cls))
            f.write('\n')
        except Exception as e:
            print(str(e))
