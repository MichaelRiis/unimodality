from __future__ import division
import os.path
import numpy as np
import numpy.random as rndm
import scipy.linalg
from scipy.stats import multivariate_normal
import itertools

from scipy.special import gammaln
from collections import OrderedDict




#######################################################################################################
# Auxilary class
#######################################################################################################

class Noisifier(object):
    """
    This class dirties function evaluations with Gaussian noise.
    If type == 'add', then the noise is additive; for type == 'mult' the noise is multiplicative.
    sd defines the magnitude of the noise, i.e., the standard deviation of the Gaussian.
    Example: ackley_noise_addp01 = Noisifier(Ackley(3), 'add', .01)
    Obviously, with the presence of noise, the max and min may no longer be accurate.
    """
    def __init__(self, func, noise_type, level):
        assert isinstance(func, Normalizer)
        if level < 0:
            raise ValueError('Noise level must be positive, level={0}'.format(level))
        self.bounds, self.min_loc, self.fmax, self.fmin = func.bounds, func.min_loc, func.fmax, func.fmin
        self.type = noise_type
        self.level = level
        self.func = func
        self.dim = self.func.dim

    def do_evaluate(self, x):
        if self.type == 'add':
            return self.func.do_evaluate(x) + self.level * np.random.normal()
        else:
            return self.func.do_evaluate(x) * (1 + self.level * np.random.normal())

    def evaluate_clean(self, x):
        return self.func.do_evaluate(x)

class Normalizer(object):
    def __init__(self, func):
        self.func = func
        self.dim = self.func.dim
        self.us_fmin = self.func.fmin
        self.deviation = self.func.fmax-self.func.do_evaluate(np.array(func.min_loc))
        bounds_array, lengths = self.tuplebounds_2_arrays(self.func.bounds) 
        self.lengths = lengths
        self.fmin = 0.0
        self.fmax = 1.0
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.us_bounds = bounds_array
        self.min_loc = (self.func.min_loc-self.us_bounds[:,0])/self.lengths
    
    def tuplebounds_2_arrays(self, bounds):
        bounds_array = np.zeros((self.dim,2))
        lengths = np.zeros((self.dim))
        for i in range(self.dim):
            bounds_array[i,0] = bounds[i][0]
            bounds_array[i,1] = bounds[i][1]
            lengths[i] = bounds[i][1]- bounds[i][0]
        return bounds_array, lengths
        
    def do_evaluate(self, x):
        return (self.func.do_evaluate( (x*self.lengths + self.us_bounds[:,0]) ) - self.us_fmin)/self.deviation
       



def tukey_function(x):
    """ Return 1-D tukey function """
    if (-1/3 - 2*x < 0) and (-1/3 + 2*x <0):
        return 1.

    elif (x > -1/2 and -1/3 - 2*x > 0):
        return 0.5*(1 + np.cos(3*np.pi*(1/6 + x)))

    elif (-1/3 + 2*x > 0 and x <= 0.5):
        return 0.5*(1 + np.cos(3*np.pi*(-1/6 + x)))

    else:
        return 0







#######################################################################################################
# Gaussian functions
#######################################################################################################
class Gaussian(object):
    def __init__(self, dim=1, num_peaks=1, seed=0, safe_limit=0., log=False):
        np.random.seed(seed)
        self.num_peaks = num_peaks
        self.num_evals = 0
        self.dim = dim
        self.log = log
        self.weights = np.random.rand(num_peaks)+np.finfo(float).eps
        self.centers = np.random.rand(num_peaks, dim)*(1.-2.*safe_limit)+safe_limit
        og = [scipy.linalg.orth(np.random.randn(dim,dim)) for i in range(num_peaks)]
        self.variances = [np.dot(np.dot(og[i], np.diag(np.random.rand(dim)*0.9+0.1)), og[i].T) / 7. for i in range(num_peaks)]
        mins = [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)]
        ind = np.argmin( [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)] )
        self.fmin = mins[ind]
        self.min_loc = self.centers[ind,:]
        self.bounds = lzip([0] * self.dim, [1] * self.dim)

        if self.log:
            xs = [0, 1]
            self.fmax = np.max([self.do_evaluate(xi) for xi in xs])
        else:
            self.fmax = 0


    def do_evaluate(self, x):
        x = np.array(x)



        if self.log:
            return np.sum( [-self.weights[i]*scipy.stats.multivariate_normal.logpdf(x, self.centers[i,:], self.variances[i]) for i in range(self.num_peaks)] )
        else:
            return np.sum( [-self.weights[i]*scipy.stats.multivariate_normal.pdf(x, self.centers[i,:], self.variances[i]) for i in range(self.num_peaks)] )

def get_gaussian_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_gaussian_functions_of_dim(num_per_dim, i+1, num_peaks, log=False) for i in range(max_dim)]

def get_log_gaussian_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_gaussian_functions_of_dim(num_per_dim, i+1, num_peaks, log=True) for i in range(max_dim)]

def get_gaussian_functions_of_dim(num, dim=1, num_peaks=1, log=False):
    return [Gaussian(dim = dim, num_peaks=num_peaks, seed=i, log=log) for i in range(num)]


#######################################################################################################
# Student t functions
#######################################################################################################

def multivariate_student_t(X, mu, Sigma, df, log=False):    
    """ multivariate student T distribution """

    [n,d] = X.shape
    Xm = X-mu
    V = df * Sigma

    L = np.linalg.cholesky(V)

    # log det of pi*V
    logdet = 2*np.sum(np.log(np.diag(L))) + d*np.log(np.pi)

    b = np.linalg.solve(L, Xm.T)

    logz = gammaln(df/2.0 + d/2.0) - gammaln(df/2.0) - 0.5*logdet
    logp = -0.5*(df+d)*np.log(1+ np.sum(b**2))

    log_pdf = logp + logz            

    if log:
        return log_pdf
    else:
        return np.exp(log_pdf)    
    

class StudentT(object):
    def __init__(self, dim=1, num_peaks=1, seed=None, safe_limit=0., log=False):

        assert(num_peaks == 1)
        assert(dim == 1)

        if seed is not None:
            np.random.seed(seed)

        self.num_peaks = num_peaks
        self.num_evals = 0
        self.dim = dim
        self.log = log
        self.weights = np.random.rand(num_peaks)+np.finfo(float).eps
        self.centers = np.random.rand(num_peaks, dim)*(1.-2.*safe_limit)+safe_limit
        og = [scipy.linalg.orth(np.random.randn(dim,dim)) for i in range(num_peaks)]
        self.variances = [np.dot(np.dot(og[i], np.diag(np.random.rand(dim)*0.9+0.1)), og[i].T) / 7. for i in range(num_peaks)]
        mins = [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)]
        ind = np.argmin( [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)] )
        self.fmin = mins[ind]
        self.min_loc = self.centers[ind,:]

        if self.log:
            xs = [0, 1]
            self.fmax = np.max([self.do_evaluate(xi) for xi in xs])
        else:
            self.fmax = 0

        # import ipdb; ipdb.set_trace()

        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        
    def do_evaluate(self, x):
        x = np.array(x)
        if self.log:
            return np.sum( [-self.weights[i]*multivariate_student_t(np.atleast_2d(x), self.centers[i,:], self.variances[i], df=1, log=True) for i in range(self.num_peaks)] )
        else:
            return np.sum( [-self.weights[i]*multivariate_student_t(np.atleast_2d(x), self.centers[i,:], self.variances[i], df=1) for i in range(self.num_peaks)] )



def get_student_t_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_student_t_functions_of_dim(num_per_dim, i+1, num_peaks) for i in range(max_dim)]

def get_log_student_t_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_student_t_functions_of_dim(num_per_dim, i+1, num_peaks, log=True) for i in range(max_dim)]


def get_student_t_functions_of_dim(num, dim=1, num_peaks=1, log=False):
    return [StudentT(dim = dim, num_peaks=num_peaks, seed=i, log=log) for i in range(num)]



#######################################################################################################
# Tukey functions
#######################################################################################################
class Tukey(object):
    def __init__(self, dim=1, num_peaks=1, seed=None, safe_limit=0.):

        assert(dim==1)

        if seed is not None:
            np.random.seed(seed)
        self.num_peaks = num_peaks
        self.num_evals = 0
        self.dim = dim
        self.weights = np.random.rand(num_peaks)+np.finfo(float).eps
        self.centers = np.random.rand(num_peaks, dim)*(1.-2.*safe_limit)+safe_limit
        og = [scipy.linalg.orth(np.random.randn(dim,dim)) for i in range(num_peaks)]
        self.variances = [np.dot(np.dot(og[i], np.diag(np.random.rand(dim)*0.9+0.1)), og[i].T) / 7. for i in range(num_peaks)]
        mins = [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)]
        ind = np.argmin( [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)] )
        self.fmin = mins[ind]
        self.min_loc = self.centers[ind,:]
        self.fmax = 0.0
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        
    def do_evaluate(self, x):
        x = np.array(x)
        return np.sum( [-self.weights[i]*tukey_function(np.atleast_2d( 0.5*(x-self.centers[i,:])/np.sqrt(self.variances[i]))) for i in range(self.num_peaks)] )



def get_tukey_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_tukey_functions_of_dim(num_per_dim, i+1, num_peaks) for i in range(max_dim)]

def get_tukey_functions_of_dim(num, dim=1, num_peaks=1):
    return [Tukey(dim = dim, num_peaks=num_peaks, seed=i) for i in range(num)]

#######################################################################################################
# Beta
#######################################################################################################
class Beta(object):
    def __init__(self, dim=1, num_peaks=1, seed=None, safe_limit=0., log=False):

        assert(dim==1)

        if seed is not None:
            np.random.seed(seed)
        self.num_peaks = num_peaks
        self.num_evals = 0
        self.dim = dim
        self.log = log

        self.weights = np.random.rand(num_peaks)+np.finfo(float).eps

        self.alphas = 1 + np.random.poisson(5, size = (self.num_peaks, 1))
        self.betas = 1+ np.random.poisson(5, size = (self.num_peaks, 1))


        self.centers = (self.alphas - 1)/(self.alphas + self.betas - 2)[:, None]

        # self.centers = np.random.rand(num_peaks, dim)*(1.-2.*safe_limit)+safe_limit
        mins = [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)]
        ind = np.argmin( [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)] )
        self.fmin = mins[ind]
        self.min_loc = self.centers[ind,:]
        self.fmax = 0.0
        self.bounds = lzip([0] * self.dim, [1] * self.dim)

        if self.log:
            xs = [1e-6, 1-1e-6]
            self.fmax = np.max([self.do_evaluate(xi) for xi in xs])

        else:
            self.fmax = 0


        
    def do_evaluate(self, x):

        x = np.min((x, 1-1e-6))
        x = np.max((x, 1e-6))
        x = np.array(x)

        if self.log:
            return np.sum( [-self.weights[i]*scipy.stats.beta.logpdf(x, self.alphas[i], self.betas[i]) for i in range(self.num_peaks)] )
        else:
            return np.sum( [-self.weights[i]*scipy.stats.beta.pdf(x, self.alphas[i], self.betas[i]) for i in range(self.num_peaks)] )

def get_beta_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_beta_functions_of_dim(num_per_dim, i+1, num_peaks) for i in range(max_dim)]

def get_log_beta_functions(num, max_dim, num_peaks):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_beta_functions_of_dim(num_per_dim, i+1, num_peaks, log=True) for i in range(max_dim)]


def get_beta_functions_of_dim(num, dim=1, num_peaks=1, log=False):
    return [Beta(dim = dim, num_peaks=num_peaks, seed=i, log=log) for i in range(num)]



#######################################################################################################
# Auxilary functions
#######################################################################################################
test_function_dict = OrderedDict( [ ('gaussian', get_gaussian_functions),
                                    ('student_t', get_student_t_functions),
                                    ('beta', get_beta_functions),
                                    ('tukey', get_tukey_functions),
                                    ('log_gaussian', get_log_gaussian_functions),
                                    ('log_student_t', get_log_student_t_functions),
                                    ('log_beta', get_log_beta_functions) ])
                                    


def lzip(*args):
    """
    Zip, but returns zipped result as a list.
    """
    return list(zip(*args))


def function_of_dimension(funcs, dim):
    ret = []
    for func in funcs:
        try:
            ret += [func(dim)]
        except AssertionError:
            pass
    return ret



def noisify_functions(func_list, noise_level):
    if isinstance(func_list, list):
        return [noisify_functions(func, noise_level) for func in func_list]
    else:
        return Noisifier(func_list, 'add', noise_level)
    
def normalize_functions(func_list):
    if isinstance(func_list, list):
        return [normalize_functions(func) for func in func_list]
    else:
        return Normalizer(func_list)


