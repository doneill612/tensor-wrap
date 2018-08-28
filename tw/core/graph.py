# -*- coding: utf-8 -*-
import abc
import tensorflow as tf

class ComputationGraph(object, metaclass=abc.ABCMeta):

    def __init__(self, mode: str, config) -> None:
        self._mode = mode
        self._config = config
        self._graph_def = None

    def mode(self) -> str:
        return self._mode

    def assign_graph_def(self, graph_def: 'tf.Graph') -> None:
        self._graph_def = graph_def

    def graph_def(self) -> 'tf.Graph':
        return self._graph_def

    @abc.abstractmethod
    def assertions(self) -> None:
        pass

    @abc.abstractmethod
    def build_train_layer_ops(self, **params) -> None:
        pass

    @abc.abstractmethod
    def build_use_layer_ops(self, **params) -> None:
        pass
