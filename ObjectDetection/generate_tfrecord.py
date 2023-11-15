from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

import tensorflow as tf


from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

def class_text_to_int(label):
    label_dict_label2num = {
        'SystemBackIcon' : 1,
        'HomeIcon' : 2,
        'Screen' : 3,
        'RadioButtonFalse' : 4,
        'BlockWithRadioButton' : 5,
        'BlackButton' : 6,
        'Scrollbar' : 7,
        'InfoIconSmall' : 8,
        'TimeTile' : 9,
        'BlueButton' : 10,
        'xIcon' : 11,
        'ArrowLeft' : 12,
        'InfoIcon' : 13,
        'PopUpWindow' : 14,
        'DealerEntry' : 15,
        'ArrowRight' : 16,
        'RadioButtonTrue' : 17,
        'ListRight' : 18,
        'ListLeft' : 19,
        'Tile' : 20,
        'DateTile' : 21,
        'UserInput' : 22,
        'BlockWithArrow' : 23,
        'BlockWithBoxSelect' : 24,
        'OverviewEntry' : 25,
        'PenIcon' : 26,
        'SettingsWindow' : 27,
        'BoxSelectFalse' : 28,
        'TabBar' : 29,
        'ButtonWithToggle' : 30,
        'SummaryEntry' : 31,
        'MenuApp' : 32,
        'App' : 33,
        'ToggleTrue' : 34,
        'LanguageSetting' : 35,
        'Keyboard' : 36,
        'ToggleFalse' : 37,
        'SearchIcon' : 38,
        'MenuBar' : 39,
        'BoxSelectTrue' : 40,
        'TrashCanIcon' : 41
    }
    return label_dict_label2num[label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # print("Encoded JPG:", type(encoded_jpg))

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.compat.v1.app.run()

# commands:
# python generate_tfrecord.py --csv_input=Outputs/train_1024x576.csv --image_dir=Fullscreen_Screenshots_1024x576 --output_path=Outputs/train_1024x576.record
# python generate_tfrecord.py --csv_input=Outputs/test_1024x576.csv --image_dir=Fullscreen_Screenshots_1024x576 --output_path=Outputs/test_1024x576.record