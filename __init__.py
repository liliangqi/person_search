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
