from __future__ import absolute_import

from ..core import Model
from ..core import TrueDiv


def image_normalizer(value=255):
    model = Model()
    model.add(TrueDiv(value), scope='preprocessing/image_normalization')
    return model
