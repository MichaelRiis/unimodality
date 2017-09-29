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

sys.path.append('../code/')
from util import plot_with_uncertainty


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
		if(len(numbers_list) != 3):
			print('Found %d numbers in filename %s, skipping...' % (len(numbers_list), f))
			continue

		# store
		ids = np.append(ids, numbers_list[1])
		files.append(f)

	num_files = len(files)
	print('Found %d valid files in total in %s' % (num_files, directory))


	# load data
	results_X = {name: [] for name in models}
	results_Y = {name: [] for name in models}

	for f, idx in zip(files, ids):

		raw_data = np.load(join(directory, f))
		for name in models:
			results_X[name].append(raw_data['results_X'][()][name])		
			results_Y[name].append(raw_data['results_Y'][()][name])	

	return results_X, results_Y

#############################################################################################################
# Metrics as a function of iterations
#############################################################################################################
def function_value(X, Y):
	return X, Y


def best_value(X, Y):
	return X, np.minimum.accumulate(np.stack(Y), axis=1)


metrics = OrderedDict([ ('Function value', function_value),
                        ('Best value', best_value)])


#############################################################################################################
# Parse arguments
#############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--directory_base', help='results directory', default = 'results')
parser.add_argument('--dim', type=int, help='dimension', default=1)

# extract
args = parser.parse_args()
directory_base = args.directory_base
dim = args.dim



models = ['regular', 'unimodal', 'unimodal2']
function_classes = ['gaussian', 'student_t']


# check that all function classes exists for specificed dim
function_classes = [function_class for function_class in function_classes if os.path.exists('{}_{}_{}d'.format(directory_base, function_class, dim))]

print('Found data for function_classes:')
for function_class in function_classes:
	print('\t%s' % function_class)
print('\n')


#############################################################################################################
# Plot
#############################################################################################################
colors = snb.color_palette()

plt.figure()

for idx_class, function_class in enumerate(function_classes):

	# read data
	directory = '{}_{}_{}d'.format(directory_base, function_class, dim)
	results_X, results_Y = read_from_directory(directory)

	# for each metric	
	for idx_metric, metric_name in enumerate(metrics):
		plt.subplot2grid((len(function_classes), len(metrics)), (idx_class, idx_metric))

		metric_fun = metrics[metric_name]
		
		for name, color in zip(models, colors):

			X, Y = metric_fun(results_X[name], results_Y[name])

			Y_mean, Y_var = np.mean(Y, axis=0), np.var(Y, axis=0)/len(X)

			plot_with_uncertainty(np.arange(1, len(Y_mean) + 1), Y_mean, yvar=Y_var, color=color, label=name)

		plt.legend()
		plt.grid(True)
		plt.xlabel('Number of iterations')
		plt.ylabel(metric_name)
		plt.title(function_class + ' (%dD)' % dim)
plt.show()

















