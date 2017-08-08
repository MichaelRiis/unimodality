import numpy as np

from scipy.stats import norm

phi = lambda x: norm.cdf(x)
logphi = lambda x: norm.logcdf(x)
npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))

class ProbitMoments(object):
	""" Class for computation of moments of distributions of the form: int (1/Z) phi((x-m)/v)*npdf(x|mu, sigma2)dx,
		where Z is the normalization constant. """

	@classmethod
	def compute_normalization(cls, m, v, mu, sigma2, log=False):
		if log:
			return logphi((mu - m)/(v*np.sqrt(1 + sigma2/v**2)))
		else:
			return phi((mu - m)/(v*np.sqrt(1 + sigma2/v**2)))

	@classmethod
	def compute_mean_and_variance(cls, m, v, mu, sigma2, return_normalizer=False):
		z = (mu - m)/(v*np.sqrt(1 + sigma2/v**2))
		Z, nz = phi(z), npdf(z, 0, 1)

		mean = mu + sigma2*nz/(Z*v*np.sqrt(1 + sigma2/v**2))
		var = sigma2 - sigma2**2*nz*(z + nz/Z)/((v**2 + sigma2)*Z)

		if not return_normalizer:
			return mean, var
		else:
			return Z, mean, var

	@classmethod
	def compute_moments(cls, m, v, mu, sigma2, normalized=True, return_normalizer=True):
		z = (mu - m)/(v*np.sqrt(1 + sigma2/v**2))
		Z, nz = phi(z), npdf(z, 0, 1)

		mean = mu + sigma2*nz/(Z*v*np.sqrt(1 + sigma2/v**2))
		mean2 = 2*mu*mean - mu**2 + sigma2 - sigma2**2*z*nz/(Z*(v**2 + sigma2))

		if not normalized:
			mean = Z*mean 
			mean2 = Z*mean2 

		if return_normalizer:
			return Z, mean, mean2
		else:
			return mean, mean2



		