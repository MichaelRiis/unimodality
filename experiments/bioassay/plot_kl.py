import numpy as np
import pylab as plt
import seaborn as snb
import sys
import argparse
import re
import os

from collections import OrderedDict
from os import listdir
from os.path import isfile, join

sys.path.append('../../code/')
from util import plot_with_uncertainty
import test_function_base

methods = ['Regular', 'Unimodal']

#############################################################################################################
# Read all files in directory
#############################################################################################################
def read_from_directory(directory):
	all_files = [f for f in listdir(directory) if isfile(join(directory, f))]
	print('Found %d files in total in %s' % (len(all_files), directory))

	# extract ids
	ids = np.array([], dtype=int)

	files = []
	for f in all_files:
		
		numbers_list = list(map(int, re.findall('\d+', f)))

		# Check that we have two and only two number
		if(len(numbers_list) != 1):
			print('Found %d numbers in filename %s, skipping...' % (len(numbers_list), f))
			continue

		# store
		ids = np.append(ids, numbers_list[0])
		files.append(f)

	num_files = len(files)
	print('Found %d valid files in total in %s' % (num_files, directory))


	# load data
	KLs = {name: [] for name in methods}
	for f, idx in zip(files, ids):

		raw_data = np.load(join(directory, f))
		for name, values in raw_data['KLs'][()].items():	
			KLs[name].append(values)

	return KLs

#############################################################################################################
# Metrics as a function of iterations
#############################################################################################################
def function_value(X, Y):
	return X, Y


def best_value(X, Y):
	return X, np.minimum.accumulate(np.stack(Y), axis=1)


metrics = OrderedDict([ ('Function value', function_value) ]) #, ('Best value', best_value)
metric_name = 'Function value'


#############################################################################################################
# Parse arguments
#############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--directory', help='results directory', default = 'results')

# extract
args = parser.parse_args()
directory = args.directory



#############################################################################################################
# Plot
#############################################################################################################
colors = snb.color_palette()


# read data
KLs = read_from_directory(directory)

plt.figure()
for name, color in zip(methods, colors):
	X, Y = Ns = np.arange(5, 100+1, 5, dtype=int), KLs[name]
	Y_mean, Y_var = np.mean(Y, axis=0), np.var(Y, axis=0)/len(X)
	plot_with_uncertainty(np.arange(1, len(Y_mean) + 1), Y_mean, yvar=Y_var, color=color, label=name)


# plt.subplot(2, int(np.ceil(len(function_classes)/2)), 1 + idx_class)

# print( int(np.ceil(len(function_classes))/2), 1 + idx_class)

# metric_fun = metrics[metric_name]

# for name, color in zip(models, colors):

# 	X, Y = metric_fun(results_X[name], results_Y[name])

# 	Y_mean, Y_var = np.mean(Y, axis=0), np.var(Y, axis=0)/len(X)

# 	plot_with_uncertainty(np.arange(1, len(Y_mean) + 1), Y_mean, yvar=Y_var, color=color, label=name)

plt.legend()
plt.grid(True)

# if idx_class == len(function_classes)-1:
# 	plt.xlabel('Number of iterations')
# plt.ylabel(metric_name)
# plt.title(function_class + ' (%dD)' % dim)
plt.show()

















