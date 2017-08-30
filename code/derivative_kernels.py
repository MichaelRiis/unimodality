import numpy as np 
import scipy as sc
import pylab as plt
import time


# kernel functions
cov_fun_base = lambda x, y, k1, k2: k1**2*np.exp(-0.5*np.sum((x[:, :, None]-y[None, :, :])**2, axis=1)/k2**2)
cov_fun0 = lambda x, y, k1, k2: cov_fun_base(x, y, k1, k2)

def cov_fun1(t2, t, g, k1, k2):
    ### Implements cov[df^i/dx^i_g, f^j] ###

    # C = np.zeros((len(t2), len(t)))

    # for i in range(len(t2)):
    #     for j in range(len(t)):
    #         C[i,j] = cov_fun0(t2[i, None], t[j, None].T)*(-(t2[i, g]-t[j, g])/k2**2)

    C = cov_fun_base(t2, t.T, k1, k2)*(-(t2[:, g, None] - t[:, g, None].T)/k2**2)

    return C


def cov_fun2(t2a, t2b, g, h, k1, k2):
    ### Implements cov[df^i/dx^i_g, df^j/dx^j_h]

    # C = np.zeros((len(t2a), len(t2b)))

    # for i in range(len(t2a)):
        # for j in range(len(t2b)):
            # C[i,j] = cov_fun0(t2a[i, None], t2b[j, None].T) *(1.0*(g==h) - (t2a[i, h] - t2b[j, h])*(t2a[i, g] - t2b[j, g])/k2**2 )/k2**2

    C =  cov_fun_base(t2a, t2b.T, k1, k2)*(1.0*(g==h) - (t2a[:, h, None] - t2b[:, h, None].T)*(t2a[:, g, None] - t2b[:, g, None].T)/k2**2 )/k2**2

    return C



def generate_joint_derivative_kernel(t, t_grad_list, k1, k2, k3 = 0, jitter = 1e-8):
    """ 
        Generates SE kernel and its derivative kernels jointly

        t:              list of input coordinates for the observation of the function
        t_grad_list:    list with input coordinates for each partial derivative of f
        k1:             magnitude parameters
        k2:             lengthscale parameter

    """

    assert(t.ndim == 2), "Wrong dimensions for t"
    for index, t_grad in enumerate(t_grad_list):

        if t_grad is not None:
            assert(t_grad.ndim == 2), "Wrong dimensions for t grad %d. t grad has dim %d x %d" % (index, t_grad.shape[0], t_grad.shape[1])
            assert(t.shape[1] == t_grad.shape[1]), "Dimensions for inputs of the partial derivative %d (%d x %d) do not match the dimensions for the inputs of the regular observations (%d x %d)" % (index, t_grad.shape[0], t_grad.shape[1], t.shape[0], t.shape[1])

        assert(t.shape[1] >= len(t_grad_list)), "Cant have more partial derivative than coordinates"

    # Dimensions
    N = len(t)
    # Ms = [len(t_grad) for t_grad in t_grad_list]
    Ms = [len(t_grad) if t_grad is not None else 0 for t_grad in t_grad_list]
    D = N + np.sum(Ms)




    # Prepare for joint kernel
    K = np.zeros((D, D))

    # Kernel for regular observations (within)
    K[:N, :N] = cov_fun0(t, t.T, k1, k2) + k3

    # Covariance between df/dx_g and f
    offset = N
    for g, M in enumerate(Ms):
        if M == 0:
            continue

        K[offset:offset + Ms[g], :N] = cov_fun1(t_grad_list[g], t, g, k1, k2)
        K[:N, offset:offset + Ms[g]] = K[offset:offset + Ms[g], :N].T
        offset += Ms[g]

    # Covariance between df/dx_g and df/dx_h
    offset_g = N
    for g, M in enumerate(Ms):
        if M == 0:
            continue

        offset_h = N

        for h in range(len(Ms)):

            if Ms[h] == 0:
                continue

            K[offset_g:offset_g + Ms[g], offset_h:offset_h + Ms[h]] = cov_fun2(t_grad_list[g], t_grad_list[h], g, h, k1, k2)
            offset_h += Ms[h]

        offset_g += Ms[g]


    # jitter
    K += jitter*np.identity(D)

    return K


if __name__ == "__main__":


    np.random.seed(100)

    # Kernel hyperparameters
    k1, k2 = 2., 10.
    
    # Dimensions etc.
    num_points_per_dim_f = 100
    num_points_per_dim_fprime = 50
    dim = 2
    x_min, x_max = -8, 8

    # Generate coordinates for function observations
    coord_grids = np.meshgrid(*[np.linspace(x_min, x_max, num_points_per_dim_f) for i in range(dim)])
    t = np.column_stack([grid.ravel() for grid in coord_grids])

    # Generate coordinates for derivative observations
    coord_grids = np.meshgrid(*[np.linspace(x_min, x_max, num_points_per_dim_fprime) for i in range(dim)])
    tg = np.column_stack([grid.ravel() for grid in coord_grids])

    # For each dim
    t_grad_list = [tg.copy() for i in range(dim)]

    # Create kernel
    t0 = time.time()
    print('Construction kernel...', end='')
    K = generate_joint_derivative_kernel(t, t_grad_list, k1, k2)
    assert(np.allclose(K, K.T)), "Matrix must be symmetric!"
    t1 = time.time()
    td = t1 - t0
    print('done in %4.3fs' % td)


    print('Size of Kernel is {} x {}'.format(*K.shape))

    try:

      t0 = time.time()
      print('Computing Cholesky decomposition...', end='')
      L = np.linalg.cholesky(K)
      t1 = time.time()
      td = t1 - t0
      print('done in %4.3fs' % td)

      if dim == 2:

        # sample f, fprime_1, fprime_2 jointly
        t0 = time.time()
        print('Sampling...', end='')
        sample = np.dot(L, np.random.normal(0, 1, size = len(K)))
        t1 = time.time()
        td = t1 - t0
        print('done in %4.3fs' % td)

        # split into parts
        f = sample[:num_points_per_dim_f**2].reshape((num_points_per_dim_f, num_points_per_dim_f))
        fp1 = sample[num_points_per_dim_f**2:num_points_per_dim_f**2 + num_points_per_dim_fprime**2].reshape((num_points_per_dim_fprime, num_points_per_dim_fprime))
        fp2 = sample[-num_points_per_dim_fprime**2:].reshape((num_points_per_dim_fprime, num_points_per_dim_fprime))


        # compute finite difference approximation of the gradients
        fp1_approx, fp2_approx = np.zeros_like(f), np.zeros_like(f)
        D1, D2 = fp1_approx.shape
        d = (x_max - x_min)/num_points_per_dim_f

        for i in range(fp1_approx.shape[0]):
            for j in range(fp1_approx.shape[1]):

                n, m = np.min((D1-1, i+1)), np.max((0, i-1))
                fp2_approx[i,j] = (f[n, j] - f[m, j])/(2*d)

                n, m = np.min((D2-1, j+1)), np.max((0, j-1))
                fp1_approx[i,j] = (f[i, n] - f[i, m])/(2*d)
        # plot
        plt.figure()

        xs = np.linspace(x_min, x_max, num_points_per_dim_f)
        xp = np.linspace(x_min, x_max, num_points_per_dim_fprime)


        for index, (xi, fi) in enumerate([(xs, f), (xp, fp1), (xp, fp2), (None, None), (xs, fp1_approx), (xs, fp2_approx)]):

            if fi is None:
                continue

            plt.subplot(2, 3, 1 + index)
            plt.pcolormesh(xi, xi, fi, cmap=plt.cm.RdBu_r)
            plt.grid(False)
            plt.colorbar()
            plt.axis('equal')
            plt.axis('off')

    except Exception as e:
      print('Faaaaiiiiil! Something went wrong')
      print(e)


    plt.show()