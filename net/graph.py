# -*- coding: utf-8 -*-
import abc
import tensorflow as tf

class ComputationGraph(object, metaclass=abc.ABCMeta):

    def __init__(self, mode, config) -> None:
        if mode not in ('train', 'use'):
            raise ValueError('mode must be either \'train\' or \'use\'')
        self._mode = mode
        self._config = config
        self._graph_def = None

    def assign_graph_def(self, graph_def):
        self._graph_def = graph_def

    @abc.abstractmethod
    def assertion_check(self):
        pass

    @abc.abstractmethod
    def build_train_layer_ops(self, **params):
        pass

    @abc.abstractmethod
    def build_use_layer_ops(self, **params):
        pass

    @abc.abstractmethod
    def build_train_ops(self, **params):
        pass

    @abc.abstractmethod
    def build_use_ops(self, **params):
        pass
