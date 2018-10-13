# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Misc flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from official.utils.flags._conventions import help_wrap


def define_image(data_format=True):
  """Register image specific flags.

  Args:
    data_format: Create a flag to specify image axis convention.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  if data_format:
    flags.DEFINE_enum(
        name="data_format", short_name="df", default=None,
        enum_values=["channels_first", "channels_last"],
        help=help_wrap(
            "A flag to override the data format used in the model. "
            "channels_first provides a performance boost on GPU but is not "
            "always compatible with CPU. If left unspecified, the data format "
            "will be chosen automatically based on whether TensorFlow was "
            "built for CPU or GPU."))
    key_flags.append("data_format")

  flags.DEFINE_integer('num_classes', -1, 'The number of classes.')
  flags.DEFINE_integer('num_images_per_epoch', -1, 'The number of images per epoch')
  flags.DEFINE_float('recall_factor', 1, 'Factor to give to recall loss')
  flags.DEFINE_float('weight_decay', 0.0001, 'Weight decay')
  flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate')

  return key_flags
