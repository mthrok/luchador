"""Define tf.Session wrapper class"""
from __future__ import absolute_import

import logging

import tensorflow as tf

import luchador.util
from ...base import scope
from ...base.wrapper import get_variable
from . import wrapper

__all__ = ['Session']
_LG = logging.getLogger(__name__)
# pylint: disable=no-member


def _get_full_class(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)


_TENSOR_CLASS_STR = _get_full_class(wrapper.Tensor)
_OP_CLASS_STR = _get_full_class(wrapper.Operation)


def _parse_outputs(outputs):
    outputs_ = []
    if outputs is None:
        return outputs_

    if not luchador.util.is_iteratable(outputs):
        outputs = [outputs]

    for output in outputs:
        if not (isinstance(output, wrapper.Tensor) or
                isinstance(output, wrapper.Variable)):
            raise ValueError(
                '`outputs` must be [list of] {}. Given: {}'
                .format(_TENSOR_CLASS_STR, _get_full_class(type(output))))
        outputs_.append(output.unwrap())
    return outputs_


def _parse_updates(updates):
    ret = []
    if not updates:
        return ret

    if not luchador.util.is_iteratable(updates):
        updates = [updates]

    for update in updates:
        if not isinstance(update, wrapper.Operation):
            raise ValueError(
                '`updates` must be [list of] {}. Given: {}'
                .format(_OP_CLASS_STR, _get_full_class(type(update))))

        ret.append(update.unwrap())
    return ret


def _construct_fetches(outputs, updates):
    return _parse_outputs(outputs) + _parse_updates(updates)


def _construct_feed_dict(inputs, givens):
    feed_dict = {}
    if not inputs:
        pass
    elif isinstance(inputs, dict):
        for key, value in inputs.items():
            feed_dict[key.unwrap()] = value
    elif isinstance(inputs, list):
        for key, value in inputs:
            feed_dict[key.unwrap()] = value
    else:
        raise ValueError(
            '`inputs` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(inputs)))

    if not givens:
        pass
    elif isinstance(givens, dict):
        for key, value in givens.items():
            feed_dict[key.unwrap()] = value
    elif isinstance(givens, list):
        for key, value in givens:
            feed_dict[key.unwrap()] = value
    else:
        raise ValueError(
            '`givens` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(givens)))
    return feed_dict


class Session(object):
    """Wrap Tensorflow Session class to work with luchador API"""
    def __init__(self, graph=None, config=None):
        super(Session, self).__init__()
        self.session = tf.Session('', graph, config)

    def _get_graph(self):
        return self.session.graph

    def _run(self, outputs=None, inputs=None,
             updates=None, givens=None, **_):
        outputs = outputs if outputs else []
        inputs = inputs if inputs else {}
        fetches = _construct_fetches(outputs, updates)
        feed_dict = _construct_feed_dict(inputs, givens)
        values = self.session.run(fetches, feed_dict=feed_dict)
        if luchador.util.is_iteratable(outputs):
            return values[:len(outputs)]
        return values[0]

    def _initialize(self):
        self.session.run(tf.global_variables_initializer())

    def _close(self):
        return self.session.close()

    ###########################################################################
    def _load_dataset(self, dataset, strict=True, **_):
        ops = []
        with scope.variable_scope(scope.VariableScope(reuse=True, name='')):
            for name, value in dataset.items():

                try:
                    variable = get_variable(name=name)
                    _LG.info(
                        '  Loading %-24s %10s -> %s %s',
                        value.shape, value.dtype, variable.dtype, name)
                except ValueError:
                    if strict:
                        raise
                    _LG.info('  Variable `%s` does not exist.', name)
                    continue

                src_shape, tgt_shape = value.shape, variable.shape
                if not tgt_shape == src_shape:
                    # Tensorflow's convolution filter shape is
                    #  [height, width, #in-channel, #out-channel]
                    # while that of Theano is
                    #  [#out-channel, #in-channel, height, width]
                    # we reshape the variable only when this condition is met
                    if (
                            len(tgt_shape) == len(src_shape) == 4 and
                            src_shape[2:4] == tgt_shape[:2] and  # h, w
                            src_shape[:2] == tgt_shape[:1:-1]  # channels
                    ):
                        _LG.info('    Reshaping variable: %s -> %s',
                                 src_shape, tgt_shape)
                        value = value.transpose((2, 3, 1, 0))
                        value = value[::-1, ::-1, :, :]
                    else:
                        raise ValueError(
                            'Shapes are not compatible. '
                            'Model shape: {}, Value shape: {}'
                            .format(src_shape, tgt_shape)
                        )
                ops.append(variable.unwrap().assign(value))
            updates = wrapper.Operation(tf.group(*ops, name='load_dataset'))
        self.run(updates=updates)
