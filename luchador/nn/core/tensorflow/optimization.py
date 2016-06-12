from __future__ import absolute_import

import tensorflow as tf

__all__ = ['clip_grad', 'build_train_ops']


def clip_grad(grads, norm):
    for i, (grad, var) in enumerate(grads):
        if grad is not None:
            grad = tf.clip_by_norm(grad, norm, name='clipped_gradient')
            grads[i] = (grad, var)


def build_train_ops(error_tensor, optimizer, clip_grad_norm=None):
    """Build training operation from given error tensor and optimizer"""
    with tf.variable_scope('training'):
        grads = optimizer.compute_gradients(error_tensor)
        if clip_grad_norm:
            clip_grad(grads, clip_grad_norm)
        train_ops = optimizer.apply_gradients(grads)
    return train_ops
