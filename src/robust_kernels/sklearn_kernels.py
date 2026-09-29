import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

from robust_kernels.tukey_kernel import tukey_kernel
from robust_kernels.andrews_kernel import andrews_kernel
from robust_kernels.huber_kernel import huber_kernel
from robust_kernels.cauchy_kernel import cauchy_kernel


class _SchoenbergKernel(Kernel):
    """Forces a robust kernel to be positive definite via the Schoenberg
    construction:

        K_PD(x, y) = exp(-gamma * Phi(x, y)),   gamma > 0

    where Phi(x, y) = -base_kernel_func(x, y, c). The base kernel functions
    (Huber, Cauchy, Andrews, Tukey) are all <= 0 with a maximum of 0 at
    x=y, so negating them gives a "distance-like" Phi that grows away from
    the diagonal -- the same role ||x-y||^2 plays in deriving the Gaussian
    kernel from the L2 loss.

    By Schoenberg's theorem, if Phi is conditionally negative definite
    (CND), then K_PD is positive definite for every gamma > 0. The paper
    proves Huber and Cauchy are conditionally positive definite (Props. 9,
    12), i.e. their negation is CND, so this construction is theoretically
    justified for those two. Tukey and Andrews are not covered by that
    guarantee (Tukey's PD-ness in the paper is argued differently, via
    Wendland functions; Andrews is proven not CPD at all, Prop. 6) -- so
    all four are still checked empirically rather than assumed to work.

    Subclasses set ``base_kernel_func``.
    """

    base_kernel_func = None

    def __init__(self, c=1.0, c_bounds=(1e-5, 1e5), gamma=1.0, gamma_bounds=(1e-5, 1e5)):
        self.c = c
        self.c_bounds = c_bounds
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

    @property
    def hyperparameter_c(self):
        return Hyperparameter("c", "numeric", self.c_bounds)

    @property
    def hyperparameter_gamma(self):
        return Hyperparameter("gamma", "numeric", self.gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise NotImplementedError(
                f"{type(self).__name__} does not support gradient evaluation; "
                "use GaussianProcessRegressor(optimizer=None)."
            )
        phi = -self.base_kernel_func(X, Y, c=self.c)
        return np.exp(-self.gamma * phi)

    def diag(self, X):
        return np.diag(self(X, X))

    def is_stationary(self):
        return True


class Tukey(_SchoenbergKernel):
    base_kernel_func = staticmethod(tukey_kernel)


class Andrews(_SchoenbergKernel):
    base_kernel_func = staticmethod(andrews_kernel)


class Huber(_SchoenbergKernel):
    base_kernel_func = staticmethod(huber_kernel)


class Cauchy(_SchoenbergKernel):
    base_kernel_func = staticmethod(cauchy_kernel)
