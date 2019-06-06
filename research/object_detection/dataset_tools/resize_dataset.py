import os
import argparse
from functools import partial
from multiprocessing import Pool

import cv2
from tqdm import tqdm


def make_dirs(directory):
    try:
        os.makedirs(directory)
    except:
        pass


def resize_image(path, source_root_dir, dest_root_dir, target_width, target_height):
    target_path = path.replace(source_root_dir, dest_root_dir)
    if not os.path.exists(os.path.dirname(target_path)):
        make_dirs(os.path.dirname(target_path))
    image = cv2.imread(path)
    image = cv2.resize(image, (target_width, target_height))
    cv2.imwrite(target_path, image)


def main(args):
    files = [os.path.join(args.source_root_dir, f) for f in os.listdir(args.source_root_dir)]
    func = partial(resize_image,
                   source_root_dir=args.source_root_dir,
                   dest_root_dir=args.dest_root_dir,
                   target_width=args.target_width,
                   target_height=args.target_height)
    with Pool(args.processes) as p:
        for _ in tqdm(p.imap(func, files), total=len(files), desc='Resizing'):
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source_root_dir', required=True)
    p.add_argument('--dest_root_dir', required=True)
    p.add_argument('--target_width', type=int, required=True)
    p.add_argument('--target_height', type=int, required=True)
    p.add_argument('--processes', type=int, required=True)
    main(p.parse_args())
