import numpy as np
import pylab as plt
import time


def plot_with_uncertainty(x, y, yvar=None, color='r', linestyle='-', fill=True, label=''):
  
  plt.plot(x, y, color=color, linestyle=linestyle, label=label)
  
  if not yvar is None:
    lower, upper = y - np.sqrt(yvar), y + np.sqrt(yvar)
    plt.plot(x, lower, color=color, alpha=0.5, linestyle=linestyle)
    plt.plot(x, upper, color=color, alpha=0.5, linestyle=linestyle)

    if fill:
      plt.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=0.25, linestyle=linestyle)


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array
    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
      From https://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%s() done in  %2.2fs' % (method.__name__, te-ts))
        return result

    return timed