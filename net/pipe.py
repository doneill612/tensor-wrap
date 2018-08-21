# -*- coding: utf-8 -*
import csv
import os
import tensorflow as tf
import numpy as np

def to_one_hot(label: int, one_hot_size: int) -> Tuple:
    """
    Encodes a label value into a one-hot vector.
    args:
        `label`: the label value
        `one_hot_size`: the length of the resulting one-hot vector.
                        this is equivalent to the total number of labels.
    returns:
        `one_hot_label` : a list object representing a one-hot vector.
    """
    one_hot_label = [0] * one_hot_size
    one_hot_label[label - 1] = 1
    return tuple(one_hot_label)

def csv_to_tf_record_onehot(csv_file: str, minmax: bool=True) -> None:
    """
    Reads a .csv file containing training data and converts each row
    into Example protos. The csv data is optionally minmax normalized before
    serialization. The label is converted into a one-hot label.
    Example protos are protocol buffers
    (https://www.tensorflow.org/programmers_guide/reading_data#file_formats
    see 'Standard TensorFlow format' section) which contain trainable
    feature information. The Example protos are serialized into a .tfrecords
    file. These files are in a binary format and constitute a `TFRecord` object.
    They can be accessed with a `TFRecordReader` obect which deserializes
    the Example protos and feeds the data to a tensor.
    args:
        `csv_file` : the file name of the .csv file to be converted to a
                     .tfrecords file
    """
    f = open(csv_file, 'r')
    try:
        reader = csv.reader(f)
        writer_path = str(csv_file).replace('.csv', '.tfrecords')
        with tf.python_io.TFRecordWriter(writer_path) as writer:
            for row in reader:
                row_float = list(map(float, row))
                label = to_one_hot(row_float[-1])
                data = row_float[:-1]
                if minmax:
                    max_val = np.max(data)
                    min_val = np.min(data)
                    data = [(d - min_val) / (max_val - min_val) for d in data]
                feature = dict(
                    X=tf.train.Feature(
                        float_list=tf.train.FloatList(value=data)),
                    y=tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label))
                )
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())
    except Exception as e:
        print("Exception encountered at #csv_to_tf_record_onehot")
        raise e
    finally:
        f.close()
