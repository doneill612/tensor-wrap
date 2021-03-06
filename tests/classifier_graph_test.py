# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

from tw.classifier.classifier_config import ClassifierConfig
from tw.classifier.classifier_model import Classifier
from tw.core.graph import ComputationGraph

class ClassifierUnitTest(tf.test.TestCase):

    def setUp(self):
        self.config = ClassifierConfig(layer_sizes=(2, 5, 5, 2),
                                       layer_activations=('relu',
                                                          'relu',
                                                          'linear'),
                                       learning_rate=0.05,
                                       batch_size=32,
                                       epochs=50,
                                       saved_model_path=\
                                        '../tw/sample_data/saved_models/',
                                       tf_record_training_file_paths=\
                                        '../tw/sample_data/xortrain.tfrecords',
                                       tf_record_validation_file_paths=\
                                        '../tw/sample_data/xorvalidate.tfrecords',
                                       tf_record_testing_file_paths=\
                                        '../tw/sample_data/xortest.tfrecords')
        self.model = Classifier(name="classifier_test", config=self.config)

    def testTrainGraphConstruction(self):
        self.model.build_computation_graph(ComputationGraph.TRAIN)
        self.model.train()

    def testUseGraphConstruction(self):
        self.model.build_computation_graph(ComputationGraph.USE)
        self.model.test()

if __name__ == '__main__':
    tf.test.main()
