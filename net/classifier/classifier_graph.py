# -*- coding: utf-8 -*
import net.graph
import net.pipe
import tensorflow as tf

class ClassifierGraph(net.graph.ComputationGraph):

    def __init__(self, mode, config):
        super(ClassifierGraph, self).__init__(mode, config)

    def assertion_check(self):
        """
        Asserts valid classifier configuration and valid graph construction mode.

        Args:
            `mode`      : the graph construction mode for the classifier
            `config`    : the classifier configuration
        Raises:
            `ValueError` if:
                - mode is not 'train', 'use', or 'test'
                - activation function specified by config is not 'sigmoid',
                    'relu', or 'tanh'
        """
        if mode not in ('train', 'use'):
            tf.logging.fatal('Bad config passed to graph builder.')
            raise ValueError('The config mode parameter '
                             'must be "train" or "use".')

        for a in config.layer_activations:
            if a not in ('sigmoid', 'relu', 'tanh', 'linear'):
                raise ValueError('The config activation parameter '
                                 'must be "sigmoid", "relu", '
                                 '"linear", or "tanh".')

    def build_train_layer_ops(self):
        pass

    def build_use_layer_ops(self):
        pass

    def build_train_ops(self):
        pass

    def build_use_ops(self):
        pass

    def finalize(self):
        pass

def build(mode, config: 'ClassifierConfig') -> 'ClassifierGraph':
    graph = ClassifierGraph(mode, config)
    if mode is 'train':
        graph.build_train_layer_ops()
        graph.build_train_ops()
    else:
        graph.build_use_layer_ops()
        graph.build_use_ops()
    graph.finalize()
    return graph
