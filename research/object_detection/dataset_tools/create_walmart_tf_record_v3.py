import os
import random

import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import contextlib2
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


random.seed(0)


labels = {0: b'left', 1: b'center', 2: b'right'}


def create_tf_example(example, mirror):
    path, boxes = example

    if mirror:
        mirror_path = path.replace('.png', '_mirror.png')
        if not os.path.exists(mirror_path):
            image = cv2.imread(path)
            image = np.fliplr(image)
            cv2.imwrite(mirror_path, image)
        path = mirror_path

    height, width = cv2.imread(path).shape[:2]
    filename = os.path.basename(path)  # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_image_data = fid.read()
    image_format = b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for box in boxes:
        cls = box[4]
        box = box[:4]
        if mirror:
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            box = [width - box[2], box[1], width - box[0], box[3]]
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            cls = 2 - cls
        box = [float(a) for a in box]
        box = [box[0] / float(width),
               box[1] / float(height),
               (box[0] + box[2]) / float(width),
               (box[1] + box[3]) / float(height)]
        box = [
            max(0., box[0]),
            max(0., box[1]),
            min(1., box[2]),
            min(1., box[3])
        ]
        xmins += [box[0]]
        xmaxs += [box[2]]
        ymins += [box[1]]
        ymaxs += [box[3]]
        classes_text += [labels[cls]]
        classes += [cls+1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def get_examples(path):
    content = [l.strip().split() for l in open(path)]
    examples = []
    for l in content:
        path = l[0]
        boxes = []
        for box_data in l[1:]:
            box = [int(a) for a in box_data.split(',')]
            if box[4] > -1:
                boxes += [box]
        if len(boxes) > 0:
            examples += [(path, boxes)]
    return examples


def write_records(examples, output_path, num_shards):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for index, example in tqdm(enumerate(examples), total=len(examples)):
            output_shard_index = index % num_shards
            tf_example = create_tf_example(example, mirror=False)
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
            tf_example = create_tf_example(example, mirror=True)
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def main():
    examples = get_examples(r'Z:\PalletWallmart\Train\Lists\train_list.txt')

    random.shuffle(examples)

    train_examples = examples[:2000]
    val_examples = examples[2000:]

    write_records(train_examples, r'D:\temp\Walmart\Records\walmart_train_v3.record', num_shards=10)
    write_records(val_examples, r'D:\temp\Walmart\Records\walmart_val_v3.record', num_shards=5)


if __name__ == '__main__':
    main()
