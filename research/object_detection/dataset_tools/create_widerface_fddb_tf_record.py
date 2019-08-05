import os

from tqdm import tqdm

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.dataset_tools.create_wider_face_tf_record import\
    process_list as process_widerface, create_tf_example


def process_list_fddb(path, min_size):
    content = [l.strip() for l in open(path)]
    examples = []
    i = 0
    num_filtered = 0
    while i < len(content):
        path = os.path.join(r'X:\fddb', content[i]) + '.jpg'
        i += 1
        num_boxes = int(content[i])
        i += 1
        faces = []
        for i in range(i, i + num_boxes):
            face = [float(a) for a in content[i].split()[:5]]
            major_axis_radius, minor_axis_radius, angle, center_x, center_y = face
            face = [center_x - minor_axis_radius,
                    center_y - major_axis_radius,
                    minor_axis_radius * 2,
                    major_axis_radius * 2]
            if (min_size == 0) or (face[2] > min_size and face[3] > min_size):
                faces += [face]
            else:
                num_filtered += 1

        i += 1
        examples += [(path, faces)]

        # import cv2
        # image = cv2.imread(path)
        # for f in faces:
        #     f = [int(a) for a in f]
        #     image = cv2.rectangle(image, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]), (0, 255, 0), 2)
        # cv2.imshow('', image)
        # cv2.waitKey()

    print('FDDB filtered: {}'.format(num_filtered))

    return examples


def main():
    num_shards = 5
    min_size = 10
    output_path = r'X:\IJB-C\NIST_11\tf_object_detection_api\records\wider_face_fddb_2\widerface_fddb_2_train{}.record'\
        .format('' if min_size == 0 else '_filter{}'.format(min_size))

    examples = []
    examples += process_widerface(r'X:\wider-face\wider_face_split\wider_face_train_bbx_gt.txt',
                                  output_path=None,
                                  num_shards=None,
                                  min_size=min_size,
                                  do_write=False)
    examples += process_list_fddb(r'X:\fddb\FDDB-folds\FDDB-all_ellipseList.txt', min_size=min_size)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for index, example in tqdm(enumerate(examples), total=len(examples)):
            tf_example = create_tf_example(example)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


if __name__ == '__main__':
    main()
