from __future__ import absolute_import

from luchador.nn import (
    Sequential,
    TrueDiv,
)


def image_normalizer(denom, dtype=None):
    model = Sequential()
    model.add_layer(TrueDiv(denom=denom, dtype=dtype),
                    scope='preprocessing/image_normalization')
    return model
