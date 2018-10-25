# -----------------------------------------------------
# Initial Settings for Training and Testing SIPN
#
# Author: Liangqi Li
# Creating Date: Apr 14, 2018
# Latest rectified: Oct 25, 2018
# -----------------------------------------------------
import time
import functools

import numpy as np


def clock_non_return(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - t0
        if elapsed < 60:
            trans_elap = elapsed
            unit = 'seconds'
        elif elapsed < 3600:
            trans_elap = elapsed / 60
            unit = 'minutes'
        else:
            trans_elap = elapsed / 3600
            unit = 'hours'
        print('\n' + '*' * 40)
        print('Entire process costs {:.2f} {:s}.'.format(trans_elap, unit))
        print('Done.')
    return clocked


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
