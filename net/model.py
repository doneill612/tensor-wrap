# -*- coding: utf-8 -*-
import abc

class Model(object, metaclass=abc.ABCMeta):
    """
    A TensorFlow-based neural network class model.
    """
    def __init__(self, name) -> None:
         """
        params:
            `name`      : the model name - used for saving purposes
            `session`   : the TensorFlow session to run the computation graph
            `graph`     : the TensorFlow computation graph
        """
        self._session = None
        self._graph = None
        self._name = name

    @abc.abstractmethod
    def build_computation_graph(self, mode):
        """
        Constructs the TensorFlow computation graph.
        params:
            mode : the graph building mode - either 'train' or 'use'
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
