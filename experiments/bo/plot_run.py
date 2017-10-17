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



#############################################################################################################
# Parse arguments
#############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--directory_base', help='results directory', default = 'results')
parser.add_argument('--function_class', help='function class', default = 'log_gaussian')
parser.add_argument('--idx', help='idx', type=int, default = 0)
parser.add_argument('--seed', help='seed', type=int, default = 0)
parser.add_argument('--dim', type=int, help='dimension', default=1)

# extract
args = parser.parse_args()
directory_base = args.directory_base
function_class = args.function_class
idx = args.idx
dim = args.dim
seed = args.seed

# build directory and filename
target_dir = '{}_{}_{}d'.format(directory_base, function_class, dim)
filename = '{}_{}d_idx{}_seed{}.npz'.format(function_class, dim, idx, seed)

print('Got the following arguments:')
print('\tInput directory: %s' % target_dir)
print('\tFilename: %s\n' % filename)

#############################################################################################################
# Load data
#############################################################################################################
fullpath = join(target_dir, filename)

try:
	data = np.load(fullpath)
	
	results_X = data['results_X'][()]
	results_Y = data['results_Y'][()]

except IOError as e:
	print(e)
	sys.exit()


print('Loaded data!')


#############################################################################################################
# Plot
#############################################################################################################

methods = list(results_X.keys())

for name in methods:

	X, Y = results_X[name], results_Y[name]
	plt.plot(Y, label=name)


plt.grid(True)
plt.legend()
plt.title('File: %s' % fullpath)
plt.xlabel('Iterations')
plt.show()