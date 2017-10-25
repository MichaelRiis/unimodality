import numpy as np
import pylab as plt
import seaborn as snb
import argparse
import GPy
import time
from autograd import value_and_grad, grad, hessian
from scipy.optimize import minimize

import os
from os import listdir
from os.path import isfile, join


import sys
sys.path.append('../../code')
import bioassay1d as bioassay

###################################################################################3
# Parse arguments
###################################################################################3
parser = argparse.ArgumentParser()
parser.add_argument('--target_directory', help='target directory', default = 'results_bio1d_v7')
parser.add_argument('--max_itt', type=int, help='target directory', default = 20)
parser.add_argument('--num_points', type=int, help='Number of  initial points', default = 1)
parser.add_argument('--seed', type=int, help='seed', default = 0)

# extract
args = parser.parse_args()
target_directory = args.target_directory
max_itt = args.max_itt
num_points = args.num_points
seed = args.seed


methods = {'Regular': bioassay.fit_regular, 'Unimodal': bioassay.fit_unimodal, 'Regular + mean function': bioassay.fit_regular_gauss}


#############################################################################################################
# Print experiment settings
#############################################################################################################
print(100*'-')
print('- Starting experiment')
print(100*'-')
print('Seed:\t\t%d' % seed)
print('Target dir:\t%s' % target_directory)
print('Num points:\t%s' % num_points)
print('Max itt:\t%s' % max_itt)




###################################################################################3
# Generate data initial data
###################################################################################3
np.random.seed(seed)

# sample initial points and evaluate posterior at these poins
X_init = np.random.uniform(0, 1, size = (1, 1))
Y_init = np.stack([bioassay.log_marginal_posterior_b(xi) for xi in X_init])[:, None]

# compute range for pertubation noise
Br = bioassay.B[2] - bioassay.B[1]

# evaluation metrics
TVs = {method: [] for method in methods}

# compute true log marginal posterior
lmp = np.stack([bioassay.log_marginal_posterior_b(b) for b in bioassay.B])

###################################################################################3
# Fit
###################################################################################3
t0 = time.time()
for method, fit_function in methods.items():

    # store data
    X = X_init.copy()
    Y = Y_init.copy()
    N = num_points

    # new data
    Xnew = np.zeros((0, 1))


    print(100*'*')
    print(100*'*')
    print('* starting experiment with %s' % method)
    print(100*'*')
    print(100*'*')

    for itt in range(max_itt):

        print(100*'-')
        print('- Iteration %d/%d (%s)' % (itt + 1, max_itt, method))
        print(100*'-')

        # fit model
        model = fit_function(X, Y)
        print(model)
        print('\n\n')

        # predict
        mu, var = model.predict(bioassay.B[:, None])
        density_mu = np.exp(mu + 0.5*var)
        density_var = (np.exp(var) - 1)*np.exp(2*mu + var)

        # sampling pertubation    
        pB = Br*np.random.uniform(0, 1) - 0.5*Br

        # sample new point
        idx = np.argmax(density_var)
        Xstar = np.atleast_2d(bioassay.B[idx])
        Ystar = bioassay.log_marginal_posterior_b(Xstar)

        # append
        X = np.row_stack((X, Xstar))
        Y = np.row_stack((Y, Ystar))

        # compute TV
        TVs[method].append(bioassay.compute_TV(model, lmp))

        print('%d: new point: %r, value=%5.4f, maxval=%5.4e\n\n' % (itt+1, Xstar, Ystar, np.max(density_var)))

t1 = time.time()
print('\n\n')
print(100*'*')
print(100*'*')
print('* Done run in %4.3fs' % (t1-t0))
print(100*'*')
print(100*'*')


###################################################################################3
# Save
###################################################################################3
save_dict = {   'target_directory': target_directory,
                'max_itt': max_itt,
                'num_points': num_points,
                'seed': seed,
                'TVs': TVs}

# create directory
if not os.path.exists(target_directory):
    os.makedirs(target_directory)   
    print('Created directory:\t%s' % target_directory)

# save to file
outputfile = join(target_directory, "bioassay_1d_seed%d" % (seed))
np.savez(outputfile, **save_dict)
print('Saved to file: %s.npz' % outputfile)

