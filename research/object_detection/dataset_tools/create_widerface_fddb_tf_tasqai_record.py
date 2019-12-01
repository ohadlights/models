import random
from tqdm import tqdm

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.dataset_tools.create_wider_face_tf_record import\
    process_list as process_widerface, create_tf_example
from object_detection.dataset_tools.create_widerface_fddb_tf_record import process_list_fddb


def process_list_tasqai(path, min_size):
    annotations = [l.strip().split(',') for l in open(path)]
    random.seed(0)
    annotations = random.sample(annotations, k=10000)
    examples = []
    num_filtered = 0
    for a in tqdm(annotations, desc='Collecting TasqAI faces'):
        path = a[0]
        coords = [int(a) for a in a[1:]]
        faces = []
        for i in range(0, len(coords), 4):
            face = coords[i:i+4]
            face = [face[0], face[1], face[2] - face[0], face[3] - face[1]]
            if (min_size == 0) or (face[2] > min_size and face[3] > min_size):
                faces += [face]
            else:
                num_filtered += 1
        if len(faces) > 0:
            examples += [(path, faces)]
    print('TasqAI filtered: {}'.format(num_filtered))
    return examples


def main():
    num_shards = 5
    min_size = 10
    output_path = r'X:\IJB-C\NIST_11\tf_object_detection_api\records\wider_face_fddb_tasqai\widerface_fddb_tasqai_train{}.record'\
        .format('' if min_size == 0 else '_filter{}'.format(min_size))

    examples = []
    examples += process_widerface(r'X:\wider-face\wider_face_split\wider_face_train_bbx_gt.txt',
                                  output_path=None,
                                  num_shards=None,
                                  min_size=min_size,
                                  do_write=False)
    examples += process_list_fddb(r'X:\fddb\FDDB-folds\FDDB-all_ellipseList.txt', min_size=min_size)
    examples += process_list_tasqai(r'X:\IJB-C\NIST_11\tf_object_detection_api\records\tasqai\tasqai_image_annotations_bbox_train.txt',
                                    min_size=min_size)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for index, example in tqdm(enumerate(examples), total=len(examples)):
            tf_example = create_tf_example(example)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


if __name__ == '__main__':
    main()
