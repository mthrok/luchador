from __future__ import absolute_import

from .core import TFModel
from .layer import TrueDiv

__all__ = ['make_image_normalizer']


def make_image_normalizer(value=255):
    model = TFModel()
    model.add(TrueDiv(value, scope='preprocessing/image_normalization'))
    return model
