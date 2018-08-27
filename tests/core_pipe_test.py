# -*- coding: utf-8 -*-
import tensorflow as tf

import tw.pipe

class PipeUnitTest(tf.test.TestCase):

    def setUp(self):
        self.training_path = '../sample_data/xortrain.csv'
        self.validation_path = '../sample_data/xorvalidate.csv'
        self.testing_path = '../sample_data/xortest.csv'

    def testCsvToTfrecord(self):
        tw.pipe.csv_to_tf_record_onehot(csv_file=self.validation_path, minmax=False)
        tw.pipe.csv_to_tf_record_onehot(csv_file=self.training_path, minmax=False)
        tw.pipe.csv_to_tf_record_onehot(csv_file=self.testing_path, minmax=False)

if __name__ == '__main__':
    tf.test.main()
