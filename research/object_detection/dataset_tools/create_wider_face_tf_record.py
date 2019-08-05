import os

import cv2
from tqdm import tqdm
import tensorflow as tf

import contextlib2
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


total_faces = 0
num_filtered = 0


def create_tf_example(example):
    path, boxes = example

    height, width = cv2.imread(path).shape[:2]
    filename = os.path.basename(path)  # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_image_data = fid.read()
    image_format = b'jpeg'  # or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for box in boxes:
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
        classes_text += [b'face']
        classes += [1]

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


def process_list(path, output_path, num_shards, min_size):
    global total_faces
    global num_filtered

    content = [l.strip() for l in open(path)]
    examples = []
    i = 0
    while i < len(content):
        path = os.path.join(r'X:\wider-face\WIDER_all\images', content[i])
        i += 1
        num_boxes = int(content[i])
        i += 1
        boxes = []
        for i in range(i, i + num_boxes):
            total_faces += 1
            box = [int(a) for a in content[i].split()[:4]]
            if (min_size == 0) or (box[2] > min_size and box[3] > min_size):
                boxes += [box]
            else:
                num_filtered += 1
        i += 1
        examples += [(path, boxes)]

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for index, example in tqdm(enumerate(examples), total=len(examples)):
            tf_example = create_tf_example(example)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def main():
    min_size = 5
    process_list(r'X:\wider-face\wider_face_split\wider_face_train_bbx_gt.txt',
                 r'X:\IJB-C\NIST_11\tf_object_detection_api\records\wider_face\wider_face_train{}.record'.format('' if min_size == 0 else '_filter{}'.format(min_size)),
                 num_shards=5,
                 min_size=min_size)
    process_list(r'X:\wider-face\wider_face_split\wider_face_val_bbx_gt.txt',
                 r'X:\IJB-C\NIST_11\tf_object_detection_api\records\wider_face\wider_face_val{}.record'.format('' if min_size == 0 else '_filter{}'.format(min_size)),
                 num_shards=2,
                 min_size=min_size)
    print('filtered {}/{}'.format(num_filtered, total_faces))


if __name__ == '__main__':
    main()
