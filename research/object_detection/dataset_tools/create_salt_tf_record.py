import os
import argparse
import io

from tqdm import tqdm
import PIL.Image as pil
import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util


def process_list(list_path, output_dir, images_root_dir, image_format, masks_root_dir):
    content = [l.strip().split(',') for l in open(list_path).readlines()]

    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir,
                                                      os.path.basename(list_path).replace('.csv', '.tfrecord')))

    for l in tqdm(content):
        image_id = l[0]

        image_path = os.path.join(images_root_dir, image_id)
        filename = image_id.encode()

        with tf.gfile.GFile(image_path + '.' + image_format, 'rb') as fid:
            encoded_png = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_png)
        image = pil.open(encoded_jpg_io)
        image = np.asarray(image)

        with tf.gfile.GFile(os.path.join(masks_root_dir, image_id + '.' + image_format), 'rb'):
            encoded_mask = fid.read()

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

        data = l[1].split()
        if len(data) > 0:

            xs = []
            ys = []

            for i in range(0, len(data), 2):
                first_index = int(data[i]) - 1
                last_index = int(data[i]) + int(data[i+1]) - 2

                xmin = first_index // height
                ymin = first_index - height * xmin

                xmax = last_index // height
                ymax = last_index - height * xmax

                xs += [xmin, xmax]
                ys += [ymin, ymax]

            xmin = min(xs)
            ymin = min(ys)
            xmax = max(xs)
            ymax = max(ys)

            xmins += [float(xmin) / float(width)]
            ymins += [float(ymin) / float(height)]
            xmaxs += [float(xmax) / float(width)]
            ymaxs += [float(ymax) / float(height)]

            classes_text += [b'salt']
            classes += [1]

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format.encode()),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/mask': dataset_util.bytes_feature(encoded_mask),
        }))

        writer.write(tf_example.SerializeToString())

    writer.close()


def main(args):
    process_list(list_path=args.train_list,
                 output_dir=args.output_dir,
                 images_root_dir=args.images_root_dir,
                 image_format=args.image_format,
                 masks_root_dir=args.mask_root_dir)
    process_list(list_path=args.val_list,
                 output_dir=args.output_dir,
                 images_root_dir=args.images_root_dir,
                 image_format=args.image_format,
                 masks_root_dir=args.mask_root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--val_list', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--images_root_dir', required=True)
    parser.add_argument('--mask_root_dir', required=True)
    parser.add_argument('--image_format', default='png', help='options: jpeg/png')
    main(parser.parse_args())
