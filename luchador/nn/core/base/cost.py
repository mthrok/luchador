from __future__ import absolute_import

import logging

from .layer import CopyMixin
from .utils import get_function_args

__all__ = ['SSE']

_LG = logging.getLogger(__name__)


class BaseCost(CopyMixin, object):
    """Base class for cost computation"""
    def __init__(self, args):
        """Validate args and set it as instance property. See CopyMixin"""
        super(BaseCost, self).__init__()
        self._store_args(args)

    def __call__(self, target, prediction):
        """Build cost between target and prediction"""
        return self.build(target, prediction)

    def build(self, target, prediction):
        raise NotImplementedError(
            '`build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )


class SSE(BaseCost):
    """Sum-Squared Error"""
    def __init__(self, max_delta=None, min_delta=None):
        super(SSE, self).__init__(args=get_function_args())
