import numpy as np
import pylab as plt


def plot_with_uncertainty(x, y, ystd=None, color='r', linestyle='-', fill=True, label=''):
	
	plt.plot(x, y, color=color, linestyle=linestyle, label=label)
	
	if not ystd is None:
		lower, upper = y - np.sqrt(ystd), y + np.sqrt(ystd)
		plt.plot(x, lower, color=color, alpha=0.5, linestyle=linestyle)
		plt.plot(x, upper, color=color, alpha=0.5, linestyle=linestyle)

	if fill:
		plt.fill_between(x.ravel(), lower, upper, color=color, alpha=0.25, linestyle=linestyle)

