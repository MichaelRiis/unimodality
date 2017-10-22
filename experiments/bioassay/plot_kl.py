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
import bioassay

methods = ['Regular', 'Regular + mean function', 'Unimodal']

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
	print('Found %d valid files in total in %s\n' % (num_files, directory))


	# load data

	KLs = {log_map: {method: [] for method in methods} for log_map in bioassay.log_density_maps}
	TVs = {log_map: {method: [] for method in methods} for log_map in bioassay.log_density_maps}


	for f, idx in zip(files, ids):

		raw_data = np.load(join(directory, f))
		KLf =  raw_data['KLs'][()]
		TVf =  raw_data['TVs'][()]

		for log_map in bioassay.log_density_maps:
			for method in methods:
				KLs[log_map][method].append(KLf[log_map][method])
				TVs[log_map][method].append(TVf[log_map][method])



	return KLs, TVs

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
snb.set(font_scale =0.8)


# read data
Ns = np.arange(5, 100+1, 5, dtype=int)

Y_KL, Y_TV = read_from_directory(directory)
metrics = {'KL': Y_KL, 'TV': Y_TV}


fig = plt.figure()

for idx_metric, metric_name in enumerate(metrics):
	
	for idx_log_map, log_map in enumerate(bioassay.log_density_maps):

		plt.subplot2grid((len(metrics), len(bioassay.log_density_maps)), (idx_metric, idx_log_map))
		plt.title('%s for %s' % (metric_name, log_map))

		for method, color in zip(methods, colors):

			X = Ns
			Y = metrics[metric_name][log_map][method]

			# check for inf and remove
			inf_list = [np.any(np.isinf(y)) for y in Y]
			print('%s: Found %d runs containing inf values' % (method, np.sum(inf_list)))
			Y = [y for (y, is_inf) in zip(Y, inf_list) if not is_inf]

			Y_mean, Y_var = np.mean(Y, axis=0), np.var(Y, axis=0)/len(Y)
			plot_with_uncertainty(Ns, Y_mean, yvar=Y_var, color=color, label=method)


		plt.legend()
		plt.grid(True)

		if idx_metric == 1:
			plt.xlabel('Number of samples')
		
		if idx_log_map == 0:
			plt.ylabel(metric_name)

		print('\n')

plt.show()

















