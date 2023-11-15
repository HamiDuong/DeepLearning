import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pyautogui
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
import xml.etree.ElementTree as ET

def get_label_id(label):
    label_dict_str2int = {
        'SystemBackIcon': 0 ,
        'HomeIcon': 1 ,
        'Screen': 2 ,
        'RadioButtonFalse': 3 ,
        'BlockWithRadioButton': 4 ,
        'BlackButton': 5 ,
        'Scrollbar': 6 ,
        'InfoIconSmall': 7 ,
        'TimeTile': 8 ,
        'BlueButton': 9 ,
        'xIcon': 10 ,
        'ArrowLeft': 11 ,
        'InfoIcon': 12 ,
        'PopUpWindow': 13 ,
        'DealerEntry': 14 ,
        'ArrowRight': 15 ,
        'RadioButtonTrue': 16 ,
        'ListRight': 17 ,
        'ListLeft': 18 ,
        'Tile': 19 ,
        'DateTile': 20 ,
        'UserInput': 21 ,
        'BlockWithArrow': 22 ,
        'BlockWithBoxSelect': 23 ,
        'OverviewEntry': 24 ,
        'PenIcon': 25 ,
        'SettingsWindow': 26 ,
        'BoxSelectFalse': 27 ,
        'TabBar': 28 ,
        'ButtonWithToggle': 29 ,
        'SummaryEntry': 30 ,
        'MenuApp': 31 ,
        'App': 32 ,
        'ToggleTrue': 33 ,
        'LanguageSetting': 34 ,
        'Keyboard': 35 ,
        'ToggleFalse': 36 ,
        'SearchIcon': 37 ,
        'MenuBar': 38 ,
        'BoxSelectTrue': 39 ,
        'TrashCanIcon': 40
    }

    return label_dict_str2int[label]


def create_tf_example_with_dataframe(row):
    # columns = ["height", "width","xml_path","jpg_path","encoded_image_data","bndbox_xmin","bndbox_xmax","bndbox_ymin","bndbox_ymax","label_text" ,"label_id"]

    height = row["height"]
    width = row["width"]
    xml_path = row["xml_path"]
    jpg_path = row["jpg_path"]
    xmin = row["bndbox_xmin"]
    xmax = row["bndbox_xmax"]
    ymin = row["bndbox_ymin"]
    ymax = row["bndbox_ymax"]
    label_text = row["label_text"]
    label_id = row["label_id"]
    image_format = b"jpg"

    encoded_image_data = row["encoded_image_data"]
    print(type(encoded_image_data))
    with tf.io.gfile.GFile(jpg_path, 'rb') as file:
        encoded_image_data = file.read()

    feature = {
            "image/height" : tf.train.Feature(int64_list = tf.train.Int64List(value = [height])),
            "image/width" : tf.train.Feature(int64_list = tf.train.Int64List(value = [width])),
            "image/filename" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [xml_path.encode("utf8")])),
            "image/source_id" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [xml_path.encode("utf8")])),
            "image/encoded" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [encoded_image_data])),
            "image/format" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_format])),
            "image/object/bbox/xmin" : tf.train.Feature(float_list = tf.train.FloatList(value = [xmin])),
            "image/object/bbox/xmax" : tf.train.Feature(float_list = tf.train.FloatList(value = [xmax])),
            "image/object/bbox/ymin" : tf.train.Feature(float_list = tf.train.FloatList(value = [ymin])),
            "image/object/bbox/ymax" : tf.train.Feature(float_list = tf.train.FloatList(value = [ymax])),
            "image/object/class/text" : tf.train.Feature(bytes_list = tf.train.BytesList(value = [label_text.encode("utf8")])),
            "image/object/class/label" : tf.train.Feature(int64_list = tf.train.Int64List(value = [label_id]))        
    }

    tf_example = tf.train.Example(features = tf.train.Features(feature = feature))
    return tf_example

def create_tfrecord_files_with_dataframe():
    # for index, row in df.iterrows():
    #     path = os.path.join(output_path, str(index) + ".tfrecord")
    #     with tf.io.TFRecordWriter(path) as writer:
    #         tf_example = create_tf_example_with_dataframe(row)
    #         writer.write(tf_example.SerializeToString())
    #     writer.close()

    writer = tf.io.TFRecordWriter("Outputs/train.record")
    df = pd.read_csv("Outputs/train.csv")
    for index, row in df.iterrows():
        tfexample = create_tf_example_with_dataframe(row)
        writer.write(tfexample.SerializeToString())
    writer.close()

    writer = tf.io.TFRecordWriter("Outputs/test.record")
    df = pd.read_csv("Outputs/test.csv")
    for index, row in df.iterrows():
        tfexample = create_tf_example_with_dataframe(row)
        writer.write(tfexample.SerializeToString())
    writer.close()

create_tfrecord_files_with_dataframe()