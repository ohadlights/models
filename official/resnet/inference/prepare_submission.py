"""
image_id,labels
2b2b327132556c767a736b3d,/m/0sgh53y /m/0g4cd0
2b2b394755692f303963553d,/m/0sgh70d /m/0g44ag
etc
"""

import os

import cv2
from tqdm import tqdm
import numpy as np

from official.resnet.inference.common import get_parser, get_classes_desc
from official.resnet.inference.model_inference import ModelInference
from official.resnet.imagenet_preprocessing import _CHANNEL_MEANS


def get_files(args):
    if args.list_path:
        return [l.split()[0] + '.jpg' for l in open(args.list_path).readlines()]
    else:
        return os.listdir(args.images_dir)


def get_ignore_classes(args):
    if not args.ignore_classes:
        return set()

    content = [l.strip().split(',') for l in open(args.classes_path, encoding='utf8').readlines()]
    class_names_to_index = {content[i][1]: i for i in range(len(content))}

    ignore_class_names = args.ignore_classes.split(',')
    ignore_class_indexes = {class_names_to_index[name] for name in ignore_class_names}

    return set(ignore_class_indexes)


def load_images(batch_paths):
    batch_images = [cv2.imread(image_path, cv2.IMREAD_UNCHANGED) for image_path in batch_paths]
    for i in range(len(batch_images)):
        batch_images[i] = batch_images[i].astype(np.float32)
        for channel in range(0, 3):
            batch_images[i][:, :, channel] -= _CHANNEL_MEANS[channel]
    return batch_images


def main(args):
    batch_size = 32

    classes_desc, num_classes = get_classes_desc(args.classes_path, map_to_id=True)

    ignore_class_indexes = get_ignore_classes(args)

    all_classifications = []
    all_classifications_conf = []

    with ModelInference(num_classes=num_classes, resnet_size=args.resnet_size, model_path=args.model_path) as model:
        files = get_files(args)
        for start_index in tqdm(range(0, len(files), batch_size)):
            batch_files = files[start_index:start_index+batch_size]
            batch_paths = [os.path.join(args.images_dir, file) for file in batch_files]
            batch_images = load_images(batch_paths)

            founds, founds_conf = model.infer(images=batch_images, threshold=args.threshold, raw_threshold=0.4)

            for i in range(0, len(batch_files)):
                file = batch_files[i]

                found = founds[i]
                found = set([index for index in found]) - ignore_class_indexes
                found = [classes_desc[index] for index in found]
                all_classifications += [(file, found)]

                found_conf = founds_conf[i]
                found_conf = [(classes_desc[a[0]], a[1]) for a in found_conf]
                all_classifications_conf += [(file, found_conf)]

    model_path = args.model_path.split('\\')

    output_path = os.path.join(args.output_dir,
                               '{}{}.txt'.format(model_path[-2], model_path[-1].replace('model.ckpt', '')))
    with open(output_path, 'w') as f:
        f.write('image_id,labels\n')
        for file, found in all_classifications:
            image_id = file.replace('.jpg', '')
            f.write('{},{}\n'.format(image_id, ' '.join(found)))

    output_path_conf = os.path.join(args.output_dir,
                                    '{}{}_conf.txt'.format(model_path[-2], model_path[-1].replace('model.ckpt', '')))
    with open(output_path_conf, 'w') as f:
        f.write('image_id,labels\n')
        for file, found in all_classifications_conf:
            image_id = file.replace('.jpg', '')
            f.write('{},{}\n'.format(image_id, ' '.join(['{}:{}'.format(a[0], a[1]) for a in found])))


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--list_path')
    parser.add_argument('--ignore_classes')
    main(parser.parse_args())
