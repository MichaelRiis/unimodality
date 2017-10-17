import numpy as np
import pylab as plt
import seaborn as snb

import sys
sys.path.append('../../code')

from importlib import reload

import test_function_base
reload(test_function_base)

dim = 1
num_functions = 5*dim
num_peaks = 1
noise_std = 0.05

np.random.seed(0)

fig = plt.figure()
for idx, name in enumerate(test_function_base.test_function_dict):

	generator_function = test_function_base.test_function_dict[name]

	print('Plotting {}'.format(name))

	functions = generator_function(num_functions, dim, num_peaks)
	functions = test_function_base.normalize_functions(functions)	
	# noisy_functions = test_function_base.noisify_functions(functions, noise_std)      

	xs = np.linspace(0, 1, 101)

	plt.subplot(2, int(np.ceil(len(test_function_base.test_function_dict)/2)), 1 + idx)
	for function_idx in range(num_functions):

		# if function_idx is not 3:
			# continue

		ys = [functions[0][function_idx].do_evaluate(xi) for xi in xs]
		plt.plot(xs, ys, alpha=0.9)
		# plt.ylim((-0.1, 1.1))



	plt.grid(True)
	plt.title(name)
plt.show()