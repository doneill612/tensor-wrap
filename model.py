# -*- coding: utf-8 -*-
import abc

class Model(object, metaclass=abc.ABCMeta):
    """
    A TensorFlow-based neural network class model.
    """
    def __init__(self, name):
         """
        params:
            `name`      : the model name - used for saving purposes
            `mode`      : the computation graph building mode - either 'train' or 'use'
            `session`   : the TensorFlow session to run the computation graph
        """
        self._session = None
        self._mode = None
        self._name = name

    @abc.abstractmethod
    def build_computation_graph(self):
        """
        Constructs the TensorFlow computation graph.
        returns:
            a `tf.Graph()` object
        """
        pass

    @abc.abstractmethod
    def train(self):
        """
        Trains a network model.
        """
        pass

    @abc.abstractmethod
    def test(self):
        """
        Runs testing on this model's trained network state.
        """
        pass
