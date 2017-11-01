import numpy as np
import pylab as plt
import seaborn as snb

import sys
sys.path.append('../../code')
from util import plot_with_uncertainty


#############################################################################################################
# Load data
#############################################################################################################
raw_data = np.load('sim_bo_data.npz')

#############################################################################################################
# Plot
#############################################################################################################
colors = snb.color_palette()
snb.set(font_scale=2)

path = '../paper/figures/'
name_dict = {'regular': 'Regular', 'unimodal': 'Unimodal'}

cc = snb.color_palette()
colors = {'regular': colors[0], 'unimodal': colors[2]}

function_classes = ['student_t',  'gaussian',  'beta',  'tukey']

for idx_class, function_class in enumerate(function_classes):

	# read data
	data = raw_data[function_class][()]

	fig = plt.figure()

	for name, color in zip(data.keys(), colors):

		X, Y_mean, Y_var = data[name]

		plot_with_uncertainty(np.arange(1, len(Y_mean) + 1), Y_mean, yvar=Y_var, color=colors[name], label=name_dict[name])

	plt.legend()
	plt.grid(True)

	plt.xlabel('Number of iterations')
	plt.ylabel('Function value')
	plt.title(function_class)

	# fig.savefig(path + 'sim_' + function_class + '.png', bbox_inches='tight')

	plt.show(block=False)

















