# -----------------------------------------------------
# Initial Settings for Taining and Testing SIPN
#
# Author: Liangqi Li
# Creating Date: Apr 14, 2018
# Latest rectified: Apr 14, 2018
# -----------------------------------------------------
import time
import functools


def clock_non_return(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        eplapsed = time.time() - t0
        print('Entire process costs {:.2f} hours.'.format(eplapsed))
    return clocked
