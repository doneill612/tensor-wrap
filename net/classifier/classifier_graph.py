# -*- coding: utf-8 -*
import net.graph

class ClassifierGraph(net.graph.ComputationGraph):

    def __init__(self, mode, config):
        super(ClassifierGraph, self).__init__(mode, config)

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
