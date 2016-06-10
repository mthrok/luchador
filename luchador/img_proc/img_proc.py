from __future__ import division
from __future__ import absolute_import

import numpy as np
from PIL import Image

__all__ = ['np2pil', 'pil2np', 'scale', 'crop']


def np2pil(arr, mode='L'):
    return Image.fromarray(arr).convert(mode)


def pil2np(image, dtype=np.uint8):
    return np.array(image, dtype=dtype)


def scale(image, height, width, interp='nearest'):
    """Scale image by matching the smaller side to the given size.

    Args:
      image (PIL image)
    Returns:
      PIL image: scaled image
    """
    w0, h0 = image.size
    ratio = max(height/h0, width/w0)
    size = (int(w0 * ratio), int(h0 * ratio))
    _resample = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    return image.resize(size, resample=_resample[interp])


def crop(image, height, width, position='center'):
    """Crop image with the given position and size

    Args:
      image (PIL image)
    Returns:
      PIL image: cropped image
    """
    w0, h0 = image.size
    if position == 'center':
        ws, hs = (w0 - width) // 2, (h0 - height) // 2
    if position == 'n':
        ws, hs = (w0 - width) // 2, h0,
    if position == 'e':
        ws, hs = (w0 - width), (h0 - height) // 2
    if position == 'w':
        ws, hs = w0, (h0 - height) // 2
    if position == 's':
        ws, hs = (w0 - width) // 2, h0 - height
    else:
        ws, hs = position['left'], position['top']
    return image.crop((ws, hs, ws+width, hs+height))
