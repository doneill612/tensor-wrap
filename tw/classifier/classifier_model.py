# -*- coding: utf-8 -*-
import tw.core.model
import tw.classifier.classifier_config
import tw.classifier.classifier_graph
import tw.classifier.classifier_runner

from tw.core.graph import ComputationGraph

class Classifier(tw.core.model.Model):

    def __init__(self, name: str, config: 'ClassifierConfig') -> None:
        super(Classifier, self).__init__(name)
        if not isinstance(config, tw.classifier.classifier_config.ClassifierConfig):
            raise ValueError('The config parameter must '
                             'be a ClassifierConfig object')
        self._config = config

    def build_computation_graph(self, mode: str) -> None:
        self._graph = tw.classifier.classifier_graph.build(mode, self._config)

    def train(self) -> None:
        if self._graph is None:
            tf.logging.fatal('None-type graph')
            raise RuntimeError('Classifier model object {id} has None-type '
                               'computation graph. Must first call '
                               '#build_computation_graph.'.format(id=self))
        if self._graph.mode() is not ComputationGraph.TRAIN:
            tf.logging.fatal('Incorrect graph mode')
            raise RuntimeError('Classifier graph {id} contains use-type ops. '
                               'Must reconstruct the computation graph in '
                               '\'train\' mode.'.format(id=self._graph))
        tw.classifier.classifier_runner.train(self._graph, self._session,
                                              self._name, self._config)

    def test(self) -> None:
        if self._graph is None:

            tf.logging.fatal('None-type graph')
            raise RuntimeError('Classifier model object {id} has None-type '
                              'computation graph. Must first call '
                              '#build_computation_graph.'.format(id=self))
        if self._graph.mode() is not ComputationGraph.USE:
            tf.logging.fatal('Incorrect graph mode')
            raise RuntimeError('Classifier graph {id} contains train-type ops. '
                              'Must reconstruct the computation graph in '
                              '\'use\' mode.'.format(id=self._graph))
        tw.classifier.classifier_runner.test(self._graph, self._session,
                                             self._name, self._config)
