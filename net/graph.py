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

    @abc.abstractmethod
    def build_train_layer_ops(self):
        pass

    @abc.abstractmethod
    def build_use_layer_ops(self):
        pass

    @abc.abstractmethod
    def build_train_ops(self):
        pass

    @abc.abstractmethod
    def build_use_ops(self):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass
