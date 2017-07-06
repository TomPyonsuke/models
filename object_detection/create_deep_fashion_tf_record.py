import hashlib
import io
import logging
import os
import random
import re

import PIL.Image
import tensorflow as tf
import collections
from itertools import izip
from utils import label_map_util
from utils import dataset_util
from collections import OrderedDict

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/media/tjiang/Elements/ImageClassification/Deep_Fashion',
                    'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/media/tjiang/Elements/ImageClassification/Deep_Fashion',
                    'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path',
                    '/media/tjiang/Elements/ImageClassification/Deep_Fashion/deep_fashion_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('random_partition', False, 'Whether to use random partion other than default')
FLAGS = flags.FLAGS

def reverse_lookup(label_map_dict, idx):
    for class_text, class_idx in label_map_dict.iteritems():
        if idx == class_idx:
            return class_text
    return ''

def read_examples_list(labels_path, bboxes_path):
    example = collections.namedtuple('example', ['label', 'bbox'])
    examples_list = []
    data = {}
    with open(labels_path, 'r') as f1, open(bboxes_path, 'r') as f2:
        f1.next(); f1.next()
        f2.next(); f2.next()
        for x, y in izip(f1, f2):
            img_name, label = x.strip().split()
            _, x_min, y_min, x_max, y_max = y.strip().split()
            examples_list.append(img_name)
            data[img_name] = example(label=label, bbox=[x_min, y_min, x_max, y_max])
    return examples_list, data

cnt = OrderedDict([(i + 1, 0) for i in range(50)])

def dict_to_tf_example(data,
                       label_map_dict,
                       img_path):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  key = hashlib.sha256(encoded_jpg).hexdigest()

  width, height = image.size

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  xmin.append(float(data.bbox[0]) / width)
  ymin.append(float(data.bbox[1]) / height)
  xmax.append(float(data.bbox[2]) / width)
  ymax.append(float(data.bbox[3]) / height)
  classes.append(int(data.label))
  cnt[int(data.label)] += 1
  text = reverse_lookup(label_map_dict, int(data.label))
  classes_text.append(text.encode('utf8'))
  truncated = []
  poses = []
  difficult_obj = []


  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     examples,
                     data):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)

  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))

    image_path = os.path.join(FLAGS.data_dir, example)
    tf_example = dict_to_tf_example(data[example], label_map_dict, image_path)
    writer.write(tf_example.SerializeToString())
  writer.close()

def get_random_split(input, seed=42, train_val_ratio=0.7):
    random.seed(seed)
    random.shuffle(input)
    num_examples = len(input)
    num_train = int(train_val_ratio * num_examples)
    train_examples = input[:num_train]
    val_examples = input[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))
    return train_examples, val_examples

def get_default_split(partition_path):
    train_examples = []
    val_examples = []
    test_examples = []
    with open(partition_path) as f:
        f.next(); f.next()
        for line in f:
            name, partition = line.strip().split()
            if partition == 'train':
                train_examples.append(name)
            elif partition == 'val':
                val_examples.append(name)
            else:
                test_examples.append(name)
    return train_examples, val_examples, test_examples




# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from deep fashion dataset.')
  annotations_dir = os.path.join(data_dir, 'Anno')
  labels_path = os.path.join(annotations_dir, 'list_category_img.txt')
  bboxes_path = os.path.join(annotations_dir, 'list_bbox.txt')
  eval_dir = os.path.join(data_dir, 'Eval')
  partition_path = os.path.join(eval_dir, 'list_eval_partition.txt')

  examples_list, data = read_examples_list(labels_path, bboxes_path)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  if FLAGS.random_partition:
    train_examples, val_examples = get_random_split(examples_list)
  else:
    train_examples, val_examples, test_examples = get_default_split(partition_path)

  train_output_path = os.path.join(FLAGS.output_dir, 'deep_fashion_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'deep_fashion_val.record')
  #create_tf_record(train_output_path, label_map_dict, train_examples, data)
  create_tf_record(val_output_path, label_map_dict, test_examples, data)
  print(cnt)

if __name__ == '__main__':
  tf.app.run()