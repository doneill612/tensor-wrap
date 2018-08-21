# -*- coding: utf-8 -*
import csv
import os
import tensorflow as tf
import numpy as np

from typing import Tuple, List

def to_one_hot(label: int, one_hot_size: int) -> Tuple:
    """
    Encodes a label value into a one-hot vector.

    Args:
        `label`: the label value
        `one_hot_size`: the length of the resulting one-hot vector.
                        this is equivalent to the total number of labels.
    Returns:
        `one_hot_label` : a list object representing a one-hot vector.
    """
    one_hot_label = [0] * one_hot_size
    one_hot_label[label - 1] = 1
    return tuple(one_hot_label)

def record_batch_onehot(tf_record_file_paths: List, batch_size: int,
                        epochs: int, input_size: int,
                        output_size: int, threads: int=3,
                        shuffle: bool=True) -> Tuple:
    """
    Reads and deserializes `batch_size` Example protos from a specified list of
    files. The batch is optinally shuffled.

    Args:
        `file_list`     : a list of file paths to .tfrecords files containing
                          Example protos
        `batch_size`    : the size of the batch to pull
        `epochs`        : the duration in epochs for which to pull batches
        `input_size`    : the input layer size of the network (essentially a
                          validation parameter which ensures that the Example
                          proto features are of the appropriate length)
        `output_size`   : the output layer size of the network (essentially a
                          validation parameter which ensures that the Example
                          proto labels are of the appropriate length)
        `threads`       : the number of threads to be used when running enqueue
                          ops
        `shuffle`       : whether or not to shuffle the data-label pairs in the
                          batch to be delivered
    Returns:
        `inputs`: a batch of inputs to be used in training
        `labels`: the associated one-hot vector labels of each of the input batches
    """
    input_producer = tf.train.string_input_producer(file_list, num_epochs=epochs)
    reader = tf.TFRecordReader()
    serialized_example = reader.read(input_producer)[1]

    features = dict(
        X=tf.FixedLenFeature([input_size], tf.float32),
        y=tf.FixedLenFeature([output_size], tf.int64)
    )

    parsed = tf.parse_single_example(serialized_example, features=features)

    parsed_inputs_as_float = tf.cast(parsed['X'], tf.float32)
    parsed_labels_as_int = tf.cast(parsed['y'], tf.int64)

    if shuffle:
        inputs, labels = tf.train.shuffle_batch([parsed_inputs_as_float,
                                                 parsed_labels_as_int],
                                                batch_size=batch_size,
                                                capacity=500,
                                                num_threads=threads,
                                                allow_smaller_final_batch=True)
    else:
        inputs, labels = tf.train.batch([parsed_inputs_as_float,
                                         parsed_labels_as_int],
                                        batch_size=batch_size,
                                        capacity=500,
                                        num_threads=threads,
                                        allow_smaller_final_batch=True)
    return inputs, labels


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

    Args:
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
        tf.logging.fatal("Exception encountered at #csv_to_tf_record_onehot")
        raise e
    finally:
        f.close()
