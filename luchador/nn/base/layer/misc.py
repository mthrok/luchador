"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BaseNHWC2NCHW', 'BaseNCHW2NHWC']

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BaseNHWC2NCHW(BaseLayer):
    """Convert NCHW data from to NHWC

    Rearrange the order of axes in 4D tensor from NHWC to NCHW

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='NHWC2NCHW'):
        super(BaseNHWC2NCHW, self).__init__(name=name)


class BaseNCHW2NHWC(BaseLayer):
    """Convert NCHW data flow to NHWC

    Rearrange the order of axes in 4D tensor from NCHW to NHWC

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='NCHW2NHWC'):
        super(BaseNCHW2NHWC, self).__init__(name=name)
