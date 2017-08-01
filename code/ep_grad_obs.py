import numpy as np

from scipy.integrate import quad
from scipy.stats import norm

npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))
log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -(x-m)**2/(2*v)
phi = lambda x: norm.cdf(x)
logphi = lambda x: norm.logcdf(x)

def _predict(mu, Sigma_full, K, t, t2, t_pred, k1, k2):
	""" returns predictive mean and full covariance """
	# kernel functions
	cov_fun = lambda x, y: k1**2*np.exp(-0.5*(x-y)**2/k2**2)
	cov_fun1 = lambda x, y: -cov_fun(x,y)*(x-y)/k2**2
	cov_fun2 = lambda x, y: cov_fun(x,y)*(1 - (x-y)**2/k2**2 )/k2**2

	D, N, P = len(mu), t.shape[0], t_pred.shape[0]

	Kpp = cov_fun(t_pred, t_pred.T)
	Kpf = np.zeros((P, D))
	Kpf[:, :N] = cov_fun(t_pred, t.T)
	Kpf[:, N:] = cov_fun1(t2, t_pred.T).T

	H =  np.linalg.solve(K, Kpf.T)
	pred_mean = np.dot(H.T, mu)
	pred_cov = Kpp -  np.dot(Kpf, H) + np.dot(H.T, np.dot(Sigma_full, H))
	return pred_mean, pred_cov


def predict(mu, Sigma_full, K, t, t2, t_pred, k1, k2):

	pred_mean, pred_cov = _predict(mu, Sigma_full, K, t, t2, t_pred, k1, k2)
	pred_var = np.diag(pred_cov)
	return pred_mean, pred_var

def lppd(mu, Sigma_full, K, t, t2, t_pred, y_pred, k1, k2, sigma2 = None, per_sample=False):
	""" compute log posterior predictive density of data pairs (t_pred, y_pred) """

	pred_mean, pred_var = predict(mu, Sigma_full, K, t, t2, t_pred, k1, k2)

	if sigma2 is None:
		sigma2 = 0

	lppds = log_npdf(y_pred.ravel(), pred_mean.ravel(), pred_var.ravel() + sigma2)

	if not per_sample:
		lppds = np.sum(lppds)

	return lppds
	
	# pred_mean, pred_cov = _predict(mu, Sigma_full, K, t, t2, t_pred, k1, k2)

	# if not sigma2 is None:
	# 	pred_cov = pred_cov + sigma2*np.identity(pred_cov.shape[0])

	# if not full_cov:
	# 	pred_cov = np.diag(np.diag(pred_cov))

	# L = np.linalg.cholesky(pred_cov + jitter*np.identity(pred_cov.shape[0]))
	# b = np.linalg.solve(L, y_pred.ravel() - pred_mean)

	# return -0.5*len(y_pred.ravel())*np.log(2*np.pi) - 0.5*2*np.sum(np.log(np.diag(L))) - 0.5*np.sum(b**2)



def generate_joint_derivative_kernel(t, t2, k1, k2, jitter = 1e-8):

	assert(t.ndim == 2), "Wrong dimensions for t"
	assert(t2.ndim == 2), "Wrong dimensions for t"

	# kernel functions
	cov_fun = lambda x, y: k1**2*np.exp(-0.5*(x-y)**2/k2**2)
	cov_fun1 = lambda x, y: -cov_fun(x,y)*(x-y)/k2**2
	cov_fun2 = lambda x, y: cov_fun(x,y)*(1 - (x-y)**2/k2**2 )/k2**2

	N, M = t.shape[0], t2.shape[0]

	# Prepare for joint kernel
	K = np.zeros((N + M, N + M))

	# Kernel for regular observations
	K[:N, :N] = cov_fun(t, t.T)

	# Kernel for derivative observations
	K[N:, N:] = cov_fun2(t2, t2.T)

	# Kernel for covariance between reg. and der. observations
	K[N:, :N] = cov_fun1(t2, t.T)
	K[:N, N:] = K[N:, :N].T

	# jitter
	K += jitter*np.identity(N + M)

	return K

def update_posterior(K, eta_z, theta_z, eta_y, theta_y):
    D = K.shape[0]
    sqrt_theta = np.sqrt(theta_z + theta_y)
    G = sqrt_theta[:, None]*K
    B = np.identity(D) + G*sqrt_theta
    L = np.linalg.cholesky(B)
    V = np.linalg.solve(L, G)
    Sigma_full = K - np.dot(V.T, V)
    mu, Sigma = np.dot(Sigma_full, eta_z + eta_y), np.diag(Sigma_full)
    
    return mu, Sigma, Sigma_full, L

def compute_marginal_likelihood(N, L, mu, Sigma, eta_z, theta_z, eta_y, theta_y, z):
    sum_eta, sum_theta = eta_z + eta_y, theta_z + theta_y

    eta_cav, theta_cav = mu/Sigma - eta_z, 1./Sigma - theta_z
    m_cav, v_cav = eta_cav/theta_cav, 1./theta_cav
    b = np.linalg.solve(L, sum_eta/np.sqrt(sum_theta))

    log_s1 = np.sum(logphi(z.ravel()*m_cav[N:]/np.sqrt(1 + v_cav[N:])))
    log_s2 = 0.5*np.sum(np.log(v_cav[N:] + 1./theta_z[N:])) + 0.5*np.sum((m_cav[N:] - eta_z[N:]/theta_z[N:])**2/(v_cav[N:] + 1./theta_z[N:]))

    logdet = np.sum(np.log(np.diag(L))) - np.sum(np.log(np.sqrt(sum_theta)))
    quadterm = 0.5*np.sum(b**2)
    return log_s1 + log_s2 - 0.5*N*np.log(2*np.pi)  - logdet - quadterm
    
def EP_grad_obs(t, t2, y, z, K, sigma2, max_itt = 100):
    
    # check dimensions
    assert(t.ndim == 2), "Wrong dimensions for t"
    assert(t2.ndim == 2), "Wrong dimensions for t2"
    assert(y.ndim == 2), "Wrong dimensions for y"
    assert(z.ndim == 2), "Wrong dimensions for z"
    assert(t.shape[0] == y.shape[0]), "Dimensions for t and y do not match"
    assert(t2.shape[0] == z.shape[0]), "Dimensions for t2 and z do not match"

    N, M = y.shape[0], z.shape[0]
    D = N + M
    
    # initialize sites and global
    eta_y, theta_y = np.zeros(D), np.zeros(D)
    eta_y[:N], theta_y[:N] = y[:, 0]/sigma2, 1./sigma2
    eta_z, theta_z = np.zeros(D), 1e-10*np.ones(D)
    mu, Sigma, Sigma_full, L = update_posterior(K, eta_z, theta_z, eta_y, theta_y)
    
    for itt in range(max_itt):

        mu_old, Sigma_old = mu.copy(), Sigma.copy()

        # loop over each gradient observation
        for j in range(M):

            i = N + j

            # compute cavity
            eta_cav, theta_cav = mu[i]/Sigma[i] - eta_z[i], 1./Sigma[i] - theta_z[i]
            m_cav, v_cav = eta_cav/theta_cav, 1./theta_cav

            # setup tilted distribution & compute moments
            tilted = lambda x: phi(z[j]*x)*npdf(x, m_cav, v_cav)
            lower, upper = m_cav - 9*np.sqrt(v_cav), m_cav + 9*np.sqrt(v_cav)
            Z = quad(tilted, lower, upper)[0]
            site_m, site_m2 = quad(lambda x: x*tilted(x), lower, upper)[0]/Z, quad(lambda x: x**2*tilted(x), lower, upper)[0]/Z
            site_v = site_m2 - site_m**2

            # update site
            eta_z[i], theta_z[i] = site_m/site_v - eta_cav, 1./site_v - theta_cav

            # update joint
            mu, Sigma, Sigma_full, L = update_posterior(K, eta_z, theta_z, eta_y, theta_y)

        # compute marginal likelihood approximation
        logL = compute_marginal_likelihood(N, L, mu, Sigma, eta_z, theta_z, eta_y, theta_y, z)

        # check convergence
        print('Itt %d, -log Z = %10.9f' % (itt, -logL))
        diff_mu, diff_Sigma = np.mean((mu-mu_old)**2), np.mean((Sigma-Sigma_old)**2)
        if diff_mu < 1e-6 and diff_Sigma < 1e-6:
            print('Converged in %d iterations with diff_mu = %5.4e and diff_Sigma = %5.4e' % (itt + 1, diff_mu, diff_Sigma))
            break
            
    return mu, Sigma_full, logL
