# -*- coding: utf-8 -*-
import abc
import tensorflow as tf

class ComputationGraph(object, metaclass=abc.ABCMeta):

    def __init__(self, mode, config) -> None:
        self._mode = mode
        self._config = config
        self._graph_def = None

    def mode(self):
        return self._mode

    def assign_graph_def(self, graph_def):
        self._graph_def = graph_def

    def graph_def(self):
        return self._graph_def

    @abc.abstractmethod
    def assertions(self):
        pass

    @abc.abstractmethod
    def build_train_layer_ops(self, **params):
        pass

    @abc.abstractmethod
    def build_use_layer_ops(self, **params):
        pass
