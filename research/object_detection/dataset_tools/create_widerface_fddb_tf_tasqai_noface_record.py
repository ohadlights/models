import cv2
from tqdm import tqdm

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.dataset_tools.create_wider_face_tf_record import\
    process_list as process_widerface, create_tf_example
from object_detection.dataset_tools.create_widerface_fddb_tf_record import process_list_fddb
from object_detection.dataset_tools.create_widerface_fddb_tf_tasqai_record import process_list_tasqai


def process_list_noface(path):
    paths = [l.strip() for l in open(path)]
    examples = []
    for p in tqdm(paths):
        if cv2.imread(p) is not None:
            examples += [(p, [])]
    return examples


def main():
    num_shards = 5
    min_size = 20
    output_path = r'X:\IJB-C\NIST_11\tf_object_detection_api\records\wider_face_fddb_tasqai_noface\widerface_fddb_tasqai_noface_train{}.record'\
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
    examples += process_list_noface(r'X:\IJB-C\NIST_11\tf_object_detection_api\records\nofaces\no_faces_list_train.txt')

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for index, example in tqdm(enumerate(examples), total=len(examples)):
            tf_example = create_tf_example(example)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


if __name__ == '__main__':
    main()
