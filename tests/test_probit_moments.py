import numpy as np
import sys
import nose

sys.path.append('../code/')
from probit_moments import ProbitMoments

from scipy.stats import norm
phi = lambda x: norm.cdf(x)

M = 10 			# number of runs
N = 5000000		# number of samples to use
max_tol = 1e-2  # Acceptable tolerance

class TestProbitMoments:

	def test_normalization_constant(self):

		print('\n')
		print(100*'-')
		print('Testing computation of normalization constant')
		print(100*'-')
		
		# Generate random numbers before starting
		zs = np.random.normal(0, 1., N)

		for m in range(M):

			# sample parameters to test
			m, v = np.random.normal(0, 1), np.random.exponential(1)
			mu, sigma2 = np.random.normal(0, 1), np.random.exponential(1)

			# compute using ProbitMoments
			computation = ProbitMoments.compute_normalization(m, v, mu, sigma2)

			# compute each constant by sampling
			xs = (np.sqrt(sigma2)*zs + mu - m)/v
			sampling = np.mean(phi(xs))

			# compare and output
			print('Sampled valued: %6.5f, computed value: %6.5f, diff = %5.4f' % (sampling, computation, sampling - computation))
			assert np.abs(computation - sampling) < max_tol



	def test_mean_and_variance(self):

		print('\n')
		print(100*'-')
		print('Testing computation of mean and variance')
		print(100*'-')

		# Generate random numbers before starting
		zs = np.random.normal(0, 1., N)

		for m in range(M):

			# sample parameters to test
			m, v = np.random.normal(0, 1), np.random.exponential(1)
			mu, sigma2 = np.random.normal(0, 1), np.random.exponential(1)

			# compute using ProbitMoments
			computation_mean, computation_var = ProbitMoments.compute_mean_and_variance(m, v, mu, sigma2)

			# compute each constant by sampling
			xs = np.sqrt(sigma2)*zs + mu

			sampling_Z = np.mean(phi((xs - m)/v))
			sampling_mean = np.mean(xs*phi((xs - m)/v))/sampling_Z
			sampling_mean2 = np.mean(xs**2*phi((xs - m)/v))/sampling_Z
			sampling_var = sampling_mean2 - sampling_mean**2

			# compare and output
			print('Sampled mean: %6.5f, computed mean: %6.5f, diff = %5.4f' % (sampling_mean, computation_mean, sampling_mean - computation_mean))
			assert np.abs(computation_mean - sampling_mean) < max_tol
			
			print('Sampled var: %6.5f, computed var: %6.5f, diff = %5.4f' % (sampling_var, computation_var, sampling_var - computation_var))
			assert np.abs(computation_var - sampling_var) < max_tol


	def test_moments(self):

		print('\n')
		print(100*'-')
		print('Testing computation of first two moments')
		print(100*'-')

		# Generate random numbers before starting
		zs = np.random.normal(0, 1., N)

		for m in range(M):

			# sample parameters to test
			m, v = np.random.normal(0, 1), np.random.exponential(1)
			mu, sigma2 = np.random.normal(0, 1), np.random.exponential(1)

			# compute using ProbitMoments
			computation_mean, computation_mean2 = ProbitMoments.compute_moments(m, v, mu, sigma2, normalized=True, return_normalizer=False)

			# compute each constant by sampling
			xs = np.sqrt(sigma2)*zs + mu

			sampling_Z = np.mean(phi((xs - m)/v))
			sampling_mean = np.mean(xs*phi((xs - m)/v))/sampling_Z
			sampling_mean2 = np.mean(xs**2*phi((xs - m)/v))/sampling_Z

			# compare and output
			print('Sampled mean: %6.5f, computed mean: %6.5f, diff = %5.4f' % (sampling_mean, computation_mean, sampling_mean - computation_mean))
			assert np.abs(computation_mean - sampling_mean) < max_tol
			
			print('Sampled mean2: %6.5f, computed mean2: %6.5f, diff = %5.4f' % (sampling_mean2, computation_mean2, sampling_mean2 - computation_mean2))
			assert np.abs(computation_mean2 - sampling_mean2) < max_tol

	def test_unnormalized_moments(self):

		print('\n')
		print(100*'-')
		print('Testing computation of unnormalized first two moments')
		print(100*'-')

		# Generate random numbers before starting
		zs = np.random.normal(0, 1., N)

		for m in range(M):

			# sample parameters to test
			m, v = np.random.normal(0, 1), np.random.exponential(1)
			mu, sigma2 = np.random.normal(0, 1), np.random.exponential(1)

			# compute using ProbitMoments
			computation_mean, computation_mean2 = ProbitMoments.compute_moments(m, v, mu, sigma2, normalized=False, return_normalizer=False)

			# compute each constant by sampling
			xs = np.sqrt(sigma2)*zs + mu

			sampling_Z = np.mean(phi((xs - m)/v))
			sampling_mean = np.mean(xs*phi((xs - m)/v))
			sampling_mean2 = np.mean(xs**2*phi((xs - m)/v))

			# compare and output
			print('Sampled mean: %6.5f, computed mean: %6.5f, diff = %5.4f' % (sampling_mean, computation_mean, sampling_mean - computation_mean))
			assert np.abs(computation_mean - sampling_mean) < max_tol
			
			print('Sampled mean2: %6.5f, computed mean2: %6.5f, diff = %5.4f' % (sampling_mean2, computation_mean2, sampling_mean2 - computation_mean2))
			assert np.abs(computation_mean2 - sampling_mean2) < max_tol




	def test_unnormalized_moments_and_normalizer(self):

		print('\n')
		print(100*'-')
		print('Testing computation of unnormalized first two moments and normalizer')
		print(100*'-')

		# Generate random numbers before starting
		zs = np.random.normal(0, 1., N)

		for m in range(M):

			# sample parameters to test
			m, v = np.random.normal(0, 1), np.random.exponential(1)
			mu, sigma2 = np.random.normal(0, 1), np.random.exponential(1)

			# compute using ProbitMoments
			computation_Z, computation_mean, computation_mean2 = ProbitMoments.compute_moments(m, v, mu, sigma2, normalized=False, return_normalizer=True)

			# compute each constant by sampling
			xs = np.sqrt(sigma2)*zs + mu

			sampling_Z = np.mean(phi((xs - m)/v))
			sampling_mean = np.mean(xs*phi((xs - m)/v))
			sampling_mean2 = np.mean(xs**2*phi((xs - m)/v))

			# compare and output
			print('Sampled Z: %6.5f, computed Z: %6.5f, diff = %5.4f' % (sampling_Z, computation_Z, sampling_Z - computation_Z))
			assert np.abs(computation_Z - sampling_Z) < max_tol

			print('Sampled mean: %6.5f, computed mean: %6.5f, diff = %5.4f' % (sampling_mean, computation_mean, sampling_mean - computation_mean))
			assert np.abs(computation_mean - sampling_mean) < max_tol
			
			print('Sampled mean2: %6.5f, computed mean2: %6.5f, diff = %5.4f' % (sampling_mean2, computation_mean2, sampling_mean2 - computation_mean2))
			assert np.abs(computation_mean2 - sampling_mean2) < max_tol



