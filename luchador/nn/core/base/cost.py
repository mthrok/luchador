from __future__ import absolute_import

import logging

from .layer import BaseLayer
from .utils import get_function_args

_LG = logging.getLogger(__name__)


class MSE(BaseLayer):
    def __init__(self, clip_delta=None, clip_error=None):
        super(MSE, self).__init__(args=get_function_args())
