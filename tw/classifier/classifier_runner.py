# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('save_freq', 2,
                            'Frequency in seconds for updating the model '
                            'save file. default = 60 sec. '
                            'Use --save_freq to change.')
tf.app.flags.DEFINE_integer('update_freq', 100,
                            'Frequency in global training step interations for sending '
                            'a training update message. default = 100 steps '
                            'Use --update_freq to change.')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            'Maxmimum number of global training steps to '
                            'take in training the model. default = 100000 steps '
                            'Use --max_steps to change.')


def _graph_collection(graph, collection) -> 'tf.Operation':
    return graph.get_collection(collection)[0]

def _log(_global_step, _cost, _validation_cost=None) -> None:
    """
    Helper function to log training information.
    args:
        `_global_step`      : current global step
        `_cost`             : current training cost
        `_validation_cost`  : current validation cost
    """
    if _validation_cost:
        tf.logging.info('Global Step: %d - '
                        'Training Loss: %.3f - '
                        'Validation Loss: %.3f',
                        _global_step, _cost, _validation_cost)
    else:
        tf.logging.info('Global Step: %d - '
                        'Training Loss: %.3f',
                        _global_step, _cost)

def train(graph: 'ClassifierGraph', sess: 'tf.Session',
          name: str, config: 'ClassifierConfig') -> None:
    """
    Trains a basic classifier network by executing the nodes
    in the TensorFlow computation graph. Training is performed
    by a `Supervisor` object.

    See https://www.tensorflow.org/api_docs/python/tf/train/Supervisor

    Args:
        graph   : the classifier compound graph object
        session : the model session object
        name    : the model name used for saving
        config  : the model configuration
    """
    graph = graph.graph_def()

    global_step = _graph_collection(graph, 'global_step')
    train_op = _graph_collection(graph, 'train_op')
    cost = _graph_collection(graph, 'cost')
    v_cost = _graph_collection(graph, 'v_cost')
    training_summary_op = _graph_collection(graph, 'training_summary_op')
    validation_summary_op = _graph_collection(graph, 'validation_summary_op')
    combined_summary_op = _graph_collection(graph, 'combined_summary_op')
    combined_cost = _graph_collection(graph, 'combined_cost')

    supervisor = tf.train.Supervisor(graph=graph,
                                     logdir=config.saved_model_path,
                                     global_step=global_step,
                                     save_model_secs=FLAGS.save_freq,
                                     checkpoint_basename='{}.ckpt'.format(name),
                                     ready_op=None)
    with supervisor.managed_session() as sess:
        train_writer = tf.summary.FileWriter('../tw/sample_data/summaries',
                                             graph=sess.graph)
        validation_writer = tf.summary.FileWriter('../tw/sample_data/v_summaries',
                                                  graph=sess.graph)
        tf.logging.info('Writers established, beginning training.')

        _global_step = sess.run(global_step)
        if _global_step >= FLAGS.max_steps:
            tf.logging.info('Max steps exceeded for this model')
            return
        while _global_step < FLAGS.max_steps:
            if supervisor.should_stop():
                train_writer.close()
                validation_writer.close()
                tf.logging.info('Supervisor stopped training.')
                break
            ud_modulo = (_global_step + 1) % FLAGS.update_freq
            v_modulo = (_global_step + 1) % (5 * FLAGS.update_freq)
            if ud_modulo == 0:
                if v_modulo == 0:
                    (_global_step, _cost, _v_cost, _, summary) = \
                        sess.run([global_step, cost, v_cost, train_op,
                                  validation_summary_op])
                    _log(_global_step, _cost, _v_cost)
                    validation_writer.add_summary(summary)
                    combined = sess.run([combined_cost, combined_summary_op],
                                        feed_dict={combined_cost: _cost})[1]
                    train_writer.add_summary(combined)
                    combined = sess.run([combined_cost, combined_summary_op],
                                        feed_dict={combined_cost: _v_cost})[1]
                    validation_writer.add_summary(combined)
                else:
                    (_global_step, _cost, _, summary) = sess.run([global_step,
                                                                  cost,
                                                                  train_op,
                                                                  training_summary_op])
                    _log(_global_step, _cost)
                    train_writer.add_summary(summary)
            else:
                (_global_step, _, summary) = sess.run([global_step, train_op,
                                                       training_summary_op])
                train_writer.add_summary(summary)
        train_writer.close()
        validation_writer.close()
        supervisor.saver.save(sess, supervisor.save_path, global_step=_global_step)
        tf.logging.info('Training complete.')

def test(graph: 'ClassifierGraph', sess: 'tf.Session',
         config: 'ClassifierConfig') -> None:
    """
    Tests the accuracy of a basic classifier network by executing the nodes
    in the TensorFlow computation graph. Unlike training which is performed
    by a `Supervisor` object, testing is performed in a less complex, more
    compact `Session` object.


    Args:
        graph   : the classifier compound graph object
        session : the model session object
        config  : the model configuration
    """
    graph = graph.graph_def()
    record_count = 1
    for r in tf.python_io.tf_record_iterator(config.tf_record_testing_file_paths):
        record_count += 1
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.saved_model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        logits = _graph_collection(graph, 'logits')
        labels = _graph_collection(graph, 'labels')
        coordinator = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coordinator)
        result = []
        for _ in range(record_count + 1):
            _logits, _labels = sess.run([logits, labels])
            _logits = np.squeeze(_logits).astype(int)
            _labels = np.squeeze(_labels).astype(int)
            correct = int((_logits == _labels).all())
            result.append(correct)
            tf.logging.info('Output: {} '
                            'Expected: {} '
                            'Correct?: {}'.format(_logits, _labels,
                                                  bool(correct)))
        tf.logging.info('Testing complete.'
                        'Accuracy = {}%'.format(100.0 * sum(result) / len(result)))
