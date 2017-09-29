import numpy as np
import sys
import argparse
import time

import os
from os import listdir
from os.path import isfile, join

sys.path.append('../code/')
import test_function_base
import bayesian_optimization as bo

#############################################################################################################
# Parse arguments
#############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--function_idx', type=int,   help='Function index', default=0)
parser.add_argument('--function_class',  help='Function class', default='gaussian')
parser.add_argument('--seed', type=int, help='Seed', default=0)
parser.add_argument('--target_directory', help='Output directory', default = 'results')
parser.add_argument('--noise_std', type = float, help='Standard deviation of noise', default = 0.05)
parser.add_argument('--maxitt', type=int, help='Number of iterations', default=17)
parser.add_argument('--dim', type=int, help='dimension', default=1)

# extract
args = parser.parse_args()
function_idx, seed, target_directory = args.function_idx, args.seed, args.target_directory
noise_std = args.noise_std
maxitt = args.maxitt
function_class = args.function_class
dim = args.dim

# store settings in dict
settings_dict = {'target_directory': target_directory,
				 'seed': seed,
				 'function_idx': function_idx,
				 'noise_std': noise_std,
				 'maxitt': maxitt,
				 'function_class': function_class,
				 'dim': dim}


target_directory += "_%s_%dd" % (function_class, dim)



#############################################################################################################
# Print experiment settings
#############################################################################################################
print(100*'-')
print('- Starting experiment')
print(100*'-')
print('Function index:\t%d' % function_idx)
print('Function class:\t%s' % function_class)
print('Dim:\t\t%d' % dim)
print('Seed:\t\t%d' % seed)
print('Max itt:\t%d' % maxitt)
print('Noise std:\t%4.3e' % noise_std)
print('Target dir:\t%s\n\n' % target_directory)


#############################################################################################################
# Define test problems
#############################################################################################################
dim = 1
num_functions = 100
num_peaks = 1

# set seed
np.random.seed(seed)

if function_class == "gaussian":
	functions0 = test_function_base.get_gaussian_functions(num_functions, dim, num_peaks)
elif function_class == "student_t":
	functions0 = test_function_base.get_student_t_functions(num_functions, dim, num_peaks)
elif function_class == "tukey":
	functions0 = test_function_base.get_tukey_functions(num_functions, dim, num_peaks)
else:

	print('Function class: %s unknown!')
	raise AssertionError
functions = test_function_base.normalize_functions(functions0)	
noisy_functions = test_function_base.noisify_functions(functions, noise_std)      

initial_points = 3
num_points = initial_points + maxitt


#############################################################################################################
# Model factory
#############################################################################################################
def regular(function_idx):
	return bo.BayesianOptimization(func_id=function_idx, func=noisy_functions[0][function_idx], acquisition_function=bo.EI, max_iter=maxitt, noise=noise_std)

def unimodal(fun):
	return bo.UnimodalBayesianOptimization(func_id=function_idx, func=noisy_functions[0][function_idx], acquisition_function=bo.EI, max_iter=maxitt, noise=noise_std)

def unimodal_g(fun):
	return bo.UnimodalBayesianOptimization(func_id=function_idx, func=noisy_functions[0][function_idx], acquisition_function=bo.EI, max_iter=maxitt, noise=noise_std, g_constraints=True)


#############################################################################################################
# Run models
#############################################################################################################
models = {'regular': regular, 'unimodal': unimodal, 'unimodal2': unimodal_g}


# preallocate
results_X = {name: np.zeros((num_points, dim)) for name in models}
results_Y = {name: np.zeros((num_points, dim)) for name in models}


for name, model_constructor in models.items():

	print(100*'-')
	print('- Running model %s' % name)
	print(100*'-')
	t0 = time.time()

	# get model instance and run
	model = model_constructor(function_idx)
	results_X[name], results_Y[name] = model.optimize()

	# done
	t1 = time.time()
	print('Finished running %s in %4.3fs' % (name, t1-t0))





#############################################################################################################
# Save results
#############################################################################################################

# to be saved
save_dict = {'settings_dict': settings_dict, 'results_X': results_X, 'results_Y': results_Y}

# create directory
if not os.path.exists(target_directory):
	os.makedirs(target_directory)
	print('Created directory:\t%s' % target_directory)

# save to file
outputfile = join(target_directory, "gaussian_1d_idx%d_seed%d" % (function_idx, seed))
np.savez(outputfile, **save_dict)
print('Saved to file: %s.npz' % outputfile)
