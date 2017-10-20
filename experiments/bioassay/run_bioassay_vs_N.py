import numpy as np
import pylab as plt
import seaborn as snb
import time
import argparse
import GPy

import os
from os import listdir
from os.path import isfile, join

from scipy.stats import multivariate_normal as mvn

from importlib import reload
import bioassay
reload(bioassay)

#############################################################################################################
# Parse arguments
#############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='Seed', default=0)
parser.add_argument('--target_directory', help='Output directory', default = 'results')
parser.add_argument('--save', type=int, help='save results', default=1)
parser.add_argument('--plot', type=int, help='plot model predictions', default=0)

# extract
args = parser.parse_args()
seed, target_directory = args.seed, args.target_directory
plot = args.plot
save = args.save




#############################################################################################################
# Print experiment settings
#############################################################################################################
print(100*'-')
print('- Starting experiment')
print(100*'-')
print('Seed:\t\t%d' % seed)
print('Target dir:\t%s' % target_directory)
print('Plot:\t\t%s' % plot)
print('Save:\t\t%d\n\n' % save)


#############################################################################################################
# Run
#############################################################################################################

# define sweep range
Ns = np.arange(5, 100+1, 5, dtype=int)

# set seed
np.random.seed(seed)

# sample initial points and evaluate posterior at these poins
Nfull = np.max(Ns)
Xfull = np.random.uniform(size = (Nfull, 2))*np.array([12, 50]) + np.array([-4, -10])
Yfull = np.stack([-bioassay.log_posterior(a,b) for (a,b) in Xfull])[:, None]

# methods
methods = {'Regular': bioassay.fit_regular, 'Unimodal': bioassay.fit_unimodal, 'Regular + mean function': bioassay.fit_regular_gauss}

KLs = {log_map: {method: [] for method in methods} for log_map in bioassay.log_density_maps}
TVs = {log_map: {method: [] for method in methods} for log_map in bioassay.log_density_maps}


# KLs_mean = {method: [] for method in methods}
# KLs_mode = {method: [] for method in methods}
# KLs_median = {method: [] for method in methods}

for idx_N, N in enumerate(Ns):

	# get data subset
	X, Y = Xfull[:N, :], Yfull[:N, :]

	for method, method_fun in methods.items():

		print(100*'-')
		print('Fitting %s GP with N = %d....' % (method, N))
		print(100*'-')

		# fit model
		model = method_fun(X, Y)

		# compute KL & TV for each estimate and print results
		print('\nFitted %s GP with N = %d' % (method, N))
		
		for log_map in bioassay.log_density_maps:
			KLs[log_map][method].append(bioassay.compute_KL(model, log_map=log_map))
			TVs[log_map][method].append(bioassay.compute_TV(model, log_map=log_map))

			print('\tKL (%s):\t%4.3f' % (log_map, KLs[log_map][method][-1]))
			print('\tTV (%s):\t%4.3f\n' % (log_map, TVs[log_map][method][-1]))


#############################################################################################################
# Save
#############################################################################################################
if save:

	# store settings in dict
	settings_dict = {'target_directory': target_directory,
				 'seed': seed,
				 'Ns': Ns}


	# to be saved
	save_dict = {'settings_dict': settings_dict, 'KLs': KLs, 'TVs': TVs,  'Ns': Ns}

	# create directory
	if not os.path.exists(target_directory):
		os.makedirs(target_directory)
		print('Created directory:\t%s' % target_directory)

	# save to file
	outputfile = join(target_directory, "bioassay_kl_seed%d" % (seed))
	np.savez(outputfile, **save_dict)
	print('Saved to file: %s.npz' % outputfile)



#############################################################################################################
# Plot results?
#############################################################################################################
if plot:
	for method, values in KLs.items():
		plt.plot(Ns, values, label=method)

	plt.grid(True)
	plt.xlabel('Number of samples')
	plt.ylabel('KL')
	plt.legend()
	plt.show()

