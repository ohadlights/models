import tensorflow as tf
import numpy as np
import cv2

from official.resnet.imagenet_main import ImagenetModel


class ModelInference:
    def __init__(self, num_classes, resnet_size, model_path):
        self.model_path = model_path

        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='inputs')

        model_class = ImagenetModel(resnet_size=resnet_size, num_classes=num_classes, resnet_version=2, dropout_rate=0.0)
        logits = model_class(inputs=self.inputs, training=False)
        self.predictions = tf.nn.sigmoid(logits)

        self.sess = None

    def __enter__(self):
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, self.model_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def infer(self, images):
        image_infer = np.array([cv2.resize(image, (224, 224)) for image in images])

        result = self.sess.run(self.predictions, feed_dict={self.inputs: image_infer})

        result = np.round(result)

        found = []
        for i in range(len(result)):
            found += [[]]
            for j in range(len(result[i])):
                if result[i][j] == 1.0:
                    found[-1] += [j]

        return found
