import numpy as np
import pylab as plt
import seaborn as snb
import sys

sys.path.append('../../code/')
from util import plot_with_uncertainty


#############################################################################################################
# Load
#############################################################################################################
data = np.load('density_tv.npz')

#############################################################################################################
# Plot
#############################################################################################################
colors = snb.color_palette()
snb.set(font_scale =1.5)


names = ['Regular', 'Regular + mean function', 'Unimodal']

fig = plt.figure()

for name, color in zip(names, colors):

	itts, Y_mean, Y_var = data[name][()]

	if name=="Regular + mean function":
		style='--'
	else:
		style='-'

	plot_with_uncertainty(itts, Y_mean, yvar=Y_var, color=color, label=name, linestyle=style)


plt.legend()
plt.grid(True)
plt.xlabel('Number of iteration')

plt.ylabel('Total variation')
path = '../../../paper/figures/'
# fig.savefig(path+'bioassay_runs.png', bbox_tight='tight')


plt.show()

















