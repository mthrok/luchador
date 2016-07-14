from __future__ import absolute_import

from luchador.nn import (
    Model,
    TrueDiv,
)


def image_normalizer(value, dtype):
    model = Model()
    model.add(TrueDiv(value, dtype), scope='preprocessing/image_normalization')
    return model
