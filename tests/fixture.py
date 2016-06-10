import numpy as np


def create_image(height=210, width=160, channel=3):
    if channel:
        shape = (height, width, channel)
    else:
        shape = (height, width)
    return np.ones(shape, dtype=np.uint8)
