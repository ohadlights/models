# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train for multi-label classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import numpy as np

from official.utils.logs import logger
from official.resnet import resnet_run_loop
from official.resnet import imagenet_preprocessing
from official.resnet.imagenet_main import define_imagenet_flags, imagenet_model_fn, _NUM_CLASSES

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3

_NUM_TRAIN_FILES = 10
_NUM_IMAGES_PER_EPOCH = 100000
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'OpenImages'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'top_10_images_train_multi.tfrecord-%05d-of-%05d' % (i, _NUM_TRAIN_FILES))
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [os.path.join(data_dir, 'top_10_images_train_multi.tfrecord-00000-of-00001')]


def _parse_example_proto(example_serialized):

    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.VarLenFeature(dtype=tf.int64),
        #'image/class/text': tf.VarLenFeature(tf.string),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    label = features['image/class/label'].values

    def create_label_vector(labels):
        label_vector = np.zeros(_NUM_CLASSES, dtype=np.float32)
        for l in labels:
            label_vector[l] = 1.0
        return label_vector
    label = tf.py_func(create_label_vector, [label], tf.float32)

    xmin = tf.expand_dims([0.], 0)
    ymin = tf.expand_dims([0.], 0)
    xmax = tf.expand_dims([1.], 0)
    ymax = tf.expand_dims([1.], 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
    """Parses a record containing a training example of an image.

      The input record is parsed into a label and image, and the image is passed
      through preprocessing steps (cropping, flipping, and so on).

      Args:
        raw_record: scalar Tensor tf.string containing a serialized
          Example protocol buffer.
        is_training: A boolean denoting whether the input is for training.
        dtype: data type to use for images/features.

      Returns:
        Tuple with processed image tensor and one-hot-encoded label tensor.
      """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)

    return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None, dtype=tf.float32):
    """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=5)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    # dataset = dataset.apply(tf.contrib.data.parallel_interleave(
    #     tf.data.TFRecordDataset, cycle_length=10))

    return resnet_run_loop.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        num_gpus=num_gpus,
        examples_per_epoch=_NUM_IMAGES_PER_EPOCH if is_training else None,
        dtype=dtype
    )


###############################################################################
# Running the model
###############################################################################
def run_training(flags_obj):
    """Run ResNet ImageNet training and eval loop.

      Args:
        flags_obj: An object containing parsed flag values.
      """

    resnet_run_loop.resnet_main(
        flags_obj, imagenet_model_fn, input_fn, DATASET_NAME,
        shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_training(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_imagenet_flags()
    absl_app.run(main)
