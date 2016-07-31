from __future__ import absolute_import

from luchador.nn import (
    Model,
    TrueDiv,
)


def image_normalizer(denom, dtype=None):
    model = Model()
    model.add_layer(TrueDiv(denom=denom, dtype=dtype),
                    scope='preprocessing/image_normalization')
    return model
