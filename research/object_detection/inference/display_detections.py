import argparse

import itertools
import cv2

import tensorflow as tf
from object_detection.inference import detection_inference


def main(args):

    with tf.Session() as sess:

        input_tfrecord_paths = [
            v for v in args.input_tfrecord_paths.split(',') if v]
        tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))

        serialized_example_tensor, image_tensor = detection_inference.build_input(
            input_tfrecord_paths)

        tf.logging.info('Reading graph and building model...')
        (detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = detection_inference.build_inference_graph(
            image_tensor, args.inference_graph)

        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()

        for counter in itertools.count():

            tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10, counter)

            serialized_example, detected_boxes, detected_scores, detected_classes = tf.get_default_session().run(
                [
                    serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor
                ])

            image = sess.run(image_tensor)[0]
            h, w = image.shape[:2]
            ymin, xmin, ymax, xmax = detected_boxes[0]
            ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
            image = cv2.rectangle(image,
                                  (xmin, ymin),
                                  (xmax, ymax),
                                  (0, 255, 0),
                                  2)
            cv2.imshow('View', image[...,::-1])
            cv2.waitKey(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_tfrecord_paths', required=True)
    p.add_argument('--inference_graph', required=True)
    main(p.parse_args())
