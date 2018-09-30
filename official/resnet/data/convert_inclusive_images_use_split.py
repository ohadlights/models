import os
import argparse
import contextlib2

from tqdm import tqdm
import tensorflow as tf

from research.object_detection.utils import dataset_util
from research.object_detection.dataset_tools import tf_record_creation_util


def create_example(raw_sample, images_root_dir, image_format, class_descriptions):
    image_id = raw_sample[0]
    label_ids = raw_sample[1:]

    class_labels = []
    classes_text = []
    for label_id in label_ids:
        class_labels += [class_descriptions[label_id][0]]
        classes_text += [class_descriptions[label_id][1].encode()]

    image_path = os.path.join(images_root_dir, image_id) + '.jpg'
    filename = image_id.encode()

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    width = 320
    height = 320

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format.encode()),
        'image/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/class/label': dataset_util.int64_list_feature(class_labels),
    }))

    return tf_example


def process_list(list_path, output_dir, images_root_dir, image_format, num_shards, class_descriptions):
    content = [l.strip().split() for l in open(list_path).readlines()]

    output_filebase = os.path.join(output_dir, os.path.basename(list_path).replace('.txt', '.tfrecord'))

    with contextlib2.ExitStack() as tf_record_close_stack:

        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)

        for index, l in tqdm(enumerate(content), total=len(content)):
            tf_example = create_example(l, images_root_dir=images_root_dir, image_format=image_format, class_descriptions=class_descriptions)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def main(args):
    class_descriptions = [l.strip().split(',') for l in open(args.class_descriptions)]
    class_descriptions = {class_descriptions[i][0]: (i, class_descriptions[i][1]) for i in range(len(class_descriptions))}
    process_list(list_path=args.train_list,
                 output_dir=args.output_dir,
                 images_root_dir=args.images_root_dir,
                 image_format=args.image_format,
                 num_shards=args.num_shards,
                 class_descriptions=class_descriptions)
    process_list(list_path=args.val_list,
                 output_dir=args.output_dir,
                 images_root_dir=args.images_root_dir,
                 image_format=args.image_format,
                 num_shards=5,
                 class_descriptions=class_descriptions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--val_list', required=True)
    parser.add_argument('--class_descriptions', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--images_root_dir', required=True)
    parser.add_argument('--image_format', default='jpeg', help='options: jpeg/png')
    parser.add_argument('--num_shards', type=int, default=250)
    main(parser.parse_args())
