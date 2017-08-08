import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from probit_moments import ProbitMoments


phi = lambda x: norm.cdf(x)
npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))



def compute_moments_softinformation(mf, vf, mg, vg, n_std=6, nu2 = 1.):

    def tilted_marginalized_f(g):
        """ Return m0, m1, m2 given by
            
            m0 = int phi( phi(g) * fp )*N(fp | mf, vf)*N(g | mg, vg) df
            m1 = int fp * phi( phi(g) * fp )*N(fp | mf, vf)*N(g | mg, vg) df
            m2 = int fp**2 * phi( phi(g) * fp )*N(fp | mf, vf)*N(g | mg, vg) df
            
        """
        
        k = (2*phi(g)-1)*nu2
        z = k*mf/(np.sqrt(1 + vf*k**2))
        Z, nz = phi(z), npdf(z, 0, 1)
        
        mean = mf + k*vf*nz/(Z*np.sqrt(1 + vf*k**2))
        mean2 = 2*mf*mean - mf**2 + vf - vf**2*z*nz/(Z*(1./k**2 + vf))
        
        m0 = Z*npdf(g, mg, vg)
        m1 = m0*mean #Z*mean*npdf(g, mg, vg)
        m2 = m0*mean2 #Z*mean2*npdf(g, mg, vg)
        
        return m0, m1, m2
    
    n_std = 6.
    g_lower, g_upper = mg - n_std*np.sqrt(vg), mg + n_std*np.sqrt(vg)
    
    # normalization constant
    Z = quad(lambda g: tilted_marginalized_f(g)[0], g_lower, g_upper)[0]
    
    # Moments wrt. fp
    Ef = quad(lambda g: tilted_marginalized_f(g)[1], g_lower, g_upper)[0]/Z
    Ef2 = quad(lambda g: tilted_marginalized_f(g)[2], g_lower, g_upper)[0]/Z
    
    # Moments wrt. g
    Eg = quad(lambda g: g*tilted_marginalized_f(g)[0], g_lower, g_upper)[0]/Z
    Eg2 = quad(lambda g: g**2*tilted_marginalized_f(g)[0], g_lower, g_upper)[0]/Z
    
    return Z, Ef, Ef2, Eg, Eg2
    


def compute_moments_strict(mf, vf, mg, vg, nu2 = 1.):
    Z_fp, m1_fp, m2_fp = ProbitMoments.compute_moments(m=0, v=1./nu2, mu=mf, sigma2=vf, return_normalizer=True, normalized=False)
    Z_g, m1_g, m2_g = ProbitMoments.compute_moments(m=0, v=1, mu=mg, sigma2=vg, return_normalizer=True, normalized=False)
    Z = (1-Z_fp)*(1-Z_g) + Z_fp*Z_g

    site_fp_m = ((mf-m1_fp)*(1-Z_g) + m1_fp*Z_g)/Z
    site_fp_m2 = (((mf**2 + vf)-m2_fp)*(1-Z_g) + m2_fp*Z_g)/Z
    site_g_m = ((1-Z_fp)*(mg-m1_g) + Z_fp*m1_g)/Z
    site_g_m2 = ((1-Z_fp)*((mg**2 + vg)-m2_g) + m2_g*Z_fp)/Z

    return Z, site_fp_m, site_fp_m2, site_g_m, site_g_m2
