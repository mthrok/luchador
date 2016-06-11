from __future__ import absolute_import

import inspect


def get_function_args():
    caller_frame = inspect.currentframe().f_back
    args = inspect.getargvalues(caller_frame)[-1]
    return {key: val for key, val in args.items() if not key == 'self'}
