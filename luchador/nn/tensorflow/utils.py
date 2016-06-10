import sys
import inspect

import tensorflow as tf


_OPTIMIZERS = sorted([
    obj[1] for obj in inspect.getmembers(
        sys.modules['tensorflow'].train, inspect.isclass)
    if issubclass(obj[1], tf.train.Optimizer)
])


def get_optimizer(name, **config):
    return getattr(tf.train, name)(**config)


def get_function_args():
    caller_frame = inspect.currentframe().f_back
    args = inspect.getargvalues(caller_frame)[-1]
    return {key: val for key, val in args.items() if not key == 'self'}
