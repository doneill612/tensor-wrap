# -*- coding: utf-8 -*-
from typing import Dict, Tuple

class ClassifierConfig(object):

    __slots__ = ('layer_sizes',
                 'layer_activations',
                 'learning_rate',
                 'batch_size',
                 'epochs'
                 'saved_model_path')

    def __init__(self, layer_sizes: Tuple, layer_activations: Tuple,
                 learning_rate: float, batch_size: int, epochs: int,
                 saved_model_path: str) -> None:

        super(ClassifierConfig, self).__setattr__('layer_sizes', layer_sizes)
        super(ClassifierConfig, self).__setattr__('layer_activations',
                                                  layer_activations)
        super(ClassifierConfig, self).__setattr__('learning_rate',
                                                  learning_rate)
        super(ClassifierConfig, self).__setattr__('batch_size', batch_size)
        super(ClassifierConfig, self).__setattr__('epochs', epochs)
        super(ClassifierConfig, self).__setattr__('saved_model_path',
                                                  saved_model_path)


    def __setattr__(self, name, value) -> None:
        """
        Override __setattr__ to prevent mutability in network configuration.
        """
        exception = ('ClassifierConfig objects are immutable -- '
                     'cannot change property {prop} value '
                     'to {val}'.format(prop=name, val=value))
        raise RuntimeError(exception)
