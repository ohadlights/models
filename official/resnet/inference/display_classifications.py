import os
import argparse

import cv2
import numpy as np
import tensorflow as tf

from official.resnet.imagenet_main import ImagenetModel


def get_classes_desc(classes_path):
    content = [l.strip().split(',') for l in open(classes_path).readlines()]
    classes_desc = {i: content[i][1] for i in range(len(content))}
    return classes_desc, len(content)


def main(args):
    classes_desc, num_classes = get_classes_desc(args.classes_path)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='inputs')

    model_class = ImagenetModel(resnet_size=args.resnet_size, num_classes=num_classes, resnet_version=2)
    logits = model_class(inputs=inputs, training=False)
    predictions = tf.nn.sigmoid(logits)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, args.model_path)

        for file in os.listdir(r'\\ger\ec\proj\ha\RSG\FacePublicDatasets\OpenImages\images\train_320'):

            image = cv2.imread(os.path.join(r'\\ger\ec\proj\ha\RSG\FacePublicDatasets\OpenImages\images\train_320', file))

            image_infer = cv2.resize(image, (224, 224))
            image_infer = np.array([image_infer])

            result = sess.run(predictions, feed_dict={inputs: image_infer})[0]

            result = np.round(result)

            found = []
            for i in range(num_classes):
                if result[i] == 1.0:
                    found += [classes_desc[i]]
            print(', '.join(found))

            cv2.imshow('view', image)
            cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--resnet_size', type=int, default=50)
    parser.add_argument('--classes_path', required=True)
    main(parser.parse_args())
