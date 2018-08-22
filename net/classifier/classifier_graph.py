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

        if len(self._config.layer_activations) != len(self._config.layer_sizes) - 1:
            tf.logging.fatal('Bad config passed to graph builder.')
            raise ValueError('The layer activations tuple must be of '
                             'length len(layer_sizes) - 1.')

        for a in self._config.layer_activations:
            if a not in ('sigmoid', 'relu', 'tanh', 'linear'):
                tf.logging.fatal('Bad config passed to graph builder.')
                raise ValueError('The config activation parameter '
                                 'must be "sigmoid", "relu", '
                                 '"linear", or "tanh".')

    def build_train_layer_ops(self, **params):
        inputs = params['inputs']
        v_inputs = params['validation_inputs']
        layer_activations = params['layer_activations']
        layer_sizes = params['layer_sizes']

        for i, channels_out in enumerate(layer_sizes[1:])
            with tf.name_scope('layer_{}'.format(i + 1)):
                channels_in = int(layer_inputs.get_shape()[1])
                W = tf.Variable(tf.random_normal(
                                    shape=[channels_in, channels_out],
                                    stddev=0.50),
                                name='weights_layer{}'.format(str(i + 1)))
                b = tf.Variable(tf.constant(1.0, shape=[channels_out]),
                                name='biases_layer{}'.format(str(i + 1)))
                inputs = tf.add(tf.matmul(inputs, weights), biases)
                v_inputs = tf.add(tf.matmul(v_inputs, weights), biases)

                activation = layer_activations[i]
                inputs = activation(inputs)
                v_inputs = activation(v_inputs)

                tf.summary.histogram('weights_summary_layer{}'.format(str(i)),
                                     W, collections=['training_summaries'])
                tf.summary.histogram('biases_summary_layer{}'.format(str(i)),
                                     b, collections=['training_summaries'])
            logits = tf.identity(inputs, name='logits')
            v_logits = tf.identity(v_inputs)

            params.update(dict(logits=logits, v_logits=vlogits))
            return params


    def build_train_ops(self, **params):
        logits = params['logits']
        v_logits = params['v_logits']
        labels = params['labels']
        v_labels = params['v_labels']
        learning_rate = params['learning_rate']

        with tf.name_scope('training'):
            global_step = tf.Variable(0, trainable=False, name='global_step',
                                      dtype=tf.int64)
            tf.add_to_collection('global_step', global_step)

            combined_cost = tf.Variable(0.0)
            tf.add_to_collection('combined_cost', combined_cost)

            cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels))
            v_cost = tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits(logits=v_logits,
                                                              labels=v_labels))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            with tf.name_scope('apply_grads'):
                train_op = optimizer.minimize(cost, global_step=global_step)

            tf.add_to_collection('cost', cost)
            tf.add_to_collection('v_cost', v_cost)
            tf.add_to_collection('train_op', train_op)

            tf.summary.scalar('cost', cost, collections=['training_summaries'])
            tf.summary.scalar('v_cost', cost, collections=['validation_summaries'])

            training_summary_op = tf.summary.merge_all('training_summaries')
            validation_summary_op = tf.summary.merge_all('validation_summaries')
            combined_summary_op = tf.summary.scalar('combined_cost', combined_cost)

            tf.add_to_collection('training_summary_op', training_summary_op)
            tf.add_to_collection('validation_summary_op', validation_summary_op)
            tf.add_to_collection('combined_summary_op', combined_summary_op)

    def build_use_layer_ops(self):
        # TODO
        pass

    def build_use_ops(self):
        # TODO
        pass

def _activation_fun_from_key(a):
    if a == 'sigmoid':
        return tf.nn.sigmoid
    elif a == 'relu':
        return tf.nn.relu
    elif a == 'tanh':
        return tf.nn.tanh
    elif a == 'linear':
        return tf.identity

def build(mode, config: 'ClassifierConfig') -> 'ClassifierGraph':
    with tf.Graph().as_default() as _graph:
        graph = ClassifierGraph(mode, config)
        graph.assertion_check()
        layer_sizes = config.layer_sizes
        train_fps = config.tf_record_training_file_paths
        v_fps = config.tf_record_validation_file_paths
        learning_rate = config.learning_rate
        layer_activations = [_activation_fun_from_key(s) for s in config.layer_activations]
        if mode is 'train':
            batch_size = config.batch_size
            epochs = config.epochs
            inputs, labels = net.pipe.record_batch_onehot(train_fps,
                                                          batch_size,
                                                          epochs,
                                                          layer_sizes[0],
                                                          layer_sizes[-1])
            v_inputs, v_labels = net.pipe.record_batch_onehot(v_fps,
                                                              batch_size,
                                                              epochs,
                                                              layer_sizes[0],
                                                              layer_sizes[-1])
            inputs = tf.identity(inputs, name='inputs')
            labels = tf.identity(labels, name='labels')
            tf.add_to_collection('inputs', inputs)
            tf.add_to_collection('labels', labels)
            _inputs = tf.identity(inputs)
            _v_inputs = tf.identity(v_inputs)
            params = dict(layer_sizes=layer_sizes, inputs=_inputs,
                          validation_inputs=_v_inputs,
                          layer_activations=layer_activations,
                          learning_rate=learning_rate)
            graph.build_train_ops(**graph.build_train_layer_ops(params))
        else:
            graph.build_use_layer_ops()
            graph.build_use_ops()
        graph.assign_graph_def(_graph)
        return graph
