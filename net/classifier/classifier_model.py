# -*- coding: utf-8 -*-
import net.model
import classifier_config
import classifier_graph
import classifier_runner

class Classifier(net.model.Model):

    def __init__(self, name: str, config: 'ClassifierConfig') -> None:
        super(Classifier, self).__init__(name)
        if not isinstance(config, classifier_config.ClassifierConfig):
            raise ValueError('The config parameter must
                             'be a ClassifierConfig object')
        self._config = config

    def build_computation_graph(self, mode: str) -> None:
        self._graph = classifier_graph.build(mode, self._config)

    def train(self) -> None:
        if self._graph is None:
            raise RuntimeError('Classifier model object {id} has None-type '
                               'computation graph. Must first call '
                               '#build_computation_graph.'.format(id=self))
        if self._graph.mode is not 'train':
            raise RuntimeError('Classifier graph {id} contains use-type ops. '
                               'Must reconstruct the computation graph in '
                               '\'train\' mode.'.format(id=self._graph))
        classifier_runner.train(self._graph, self._session,
                               self._name, self._config)

    def test(self) -> None:
        if self._graph is None:
           raise RuntimeError('Classifier model object {id} has None-type '
                              'computation graph. Must first call '
                              '#build_computation_graph.'.format(id=self))
        if self._graph.mode is not 'use':
           raise RuntimeError('Classifier graph {id} contains train-type ops. '
                              'Must reconstruct the computation graph in '
                              '\'use\' mode.'.format(id=self._graph))
        classifier_runner.test(self._graph, self._session,
                               self._name, self._config)
