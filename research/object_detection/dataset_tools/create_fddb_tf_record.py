import io
import os

import cv2
from tqdm import tqdm
import tensorflow as tf

from object_detection.utils import dataset_util


def create_tf_example(example):

    path, faces = example

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

    for face in faces:
        major_axis_radius, minor_axis_radius, angle, center_x, center_y = face
        box = [center_x - minor_axis_radius,
               center_y - major_axis_radius,
               center_x + minor_axis_radius,
               center_y + major_axis_radius]
        xmins += [box[0] / float(width)]
        xmaxs += [box[2] / float(width)]
        ymins += [box[1] / float(height)]
        ymaxs += [box[3] / float(height)]
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


def process_list(path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)

    content = [l.strip() for l in open(path)]
    examples = []
    i = 0
    while i < len(content):
        path = os.path.join(r'X:\fddb', content[i]) + '.jpg'
        i += 1
        num_boxes = int(content[i])
        i += 1
        faces = []
        for i in range(i, i + num_boxes):
            faces += [[float(a) for a in content[i].split()[:5]]]
        i += 1
        examples += [(path, faces)]

    for example in tqdm(examples):
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main():
    process_list(r'X:\fddb\FDDB-folds\FDDB-all_ellipseList.txt',
                 r'X:\IJB-C\NIST_11\tf_object_detection_api\fddb\records\fddb_train.record')


if __name__ == '__main__':
    main()
