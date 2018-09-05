import os
import argparse
import io

from tqdm import tqdm
import PIL.Image as pil
import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util


def create_example(raw_sample, images_root_dir, image_format):
    image_id = raw_sample[0]
    boxes = [b.split(',') for b in raw_sample[1:]]

    image_path = os.path.join(images_root_dir, image_id)
    filename = image_id.encode()

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = pil.open(encoded_jpg_io)
    image = np.asarray(image)

    width = int(image.shape[1])
    height = int(image.shape[0])

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for box in boxes:
        xmins += [float(box[0]) / float(width)]
        ymins += [float(box[1]) / float(height)]
        xmaxs += [float(box[2]) / float(width)]
        ymaxs += [float(box[3]) / float(height)]

        classes_text += [b'face']
        classes += [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format.encode()),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def process_list(list_path, output_dir, images_root_dir, image_format, num_shards):
    content = [l.strip().split() for l in open(list_path).readlines()]

    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir,
                                                      os.path.basename(list_path).replace('.txt', '.tfrecord')))

    for l in tqdm(content):
        tf_example = create_example(l, images_root_dir=images_root_dir, image_format=image_format)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(args):
    process_list(list_path=args.train_list,
                 output_dir=args.output_dir,
                 images_root_dir=args.images_root_dir,
                 image_format=args.image_format,
                 num_shards=args.num_shards)
    process_list(list_path=args.val_list,
                 output_dir=args.output_dir,
                 images_root_dir=args.images_root_dir,
                 image_format=args.image_format,
                 num_shards=args.num_shards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--val_list', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--images_root_dir', required=True)
    parser.add_argument('--image_format', default='png', help='options: jpeg/png')
    parser.add_argument('--num_shards', default=10)
    main(parser.parse_args())
