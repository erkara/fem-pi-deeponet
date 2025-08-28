import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import numpy as npo
from scipy.integrate import dblquad



class GaussianSampler2D:
    def __init__(self, problem_config):
        self.center_range = problem_config['center_range']      # ((x_min, x_max), (y_min, y_max))
        self.sigma_range = problem_config['sigma_range']        # (sigma_min, sigma_max)
        self.domain_x = problem_config['space_domain'][0]       # (x_min, x_max)
        self.domain_y = problem_config['space_domain'][1]       # (y_min, y_max)

        # self.mode 'fixed' or 'variable', value is used in both
        # self.mode = `variable` or `fixed` means each data instance can have different
        # number of sources less than or equal to self.value
        gauss_mode = problem_config['gauss_mode']
        self.mode = gauss_mode['type']
        self.value = gauss_mode['value']

        # this is for only one experiment case
        self.fixed_center = problem_config.get('fixed_center', False)

        if self.mode not in ['fixed', 'variable']:
            raise ValueError(f"Unsupported gauss_mode type: {self.mode}")

    def sample(self, rng):
        if self.mode == 'fixed':
            num_gauss = self.value
        else:
            num_gauss = rng.integers(1, self.value + 1)

        # sample sigmas
        sigma_list = rng.uniform(self.sigma_range[0], self.sigma_range[1], num_gauss)

        # customize if needed.
        if self.fixed_center:
            if num_gauss == 3:
                x0_list = npo.array([3, 4, 7])
                y0_list = npo.array([3, 6, 4])
            if num_gauss == 1:
                x0_list = npo.array([5])
                y0_list = npo.array([5])
            if num_gauss == 2:
                x0_list = npo.array([3,7])
                y0_list = npo.array([3,5])

        else:
            x0_list = rng.uniform(self.center_range[0][0], self.center_range[0][1], num_gauss)
            y0_list = rng.uniform(self.center_range[1][0], self.center_range[1][1], num_gauss)
        return {
            'centers': list(zip(x0_list, y0_list)),
            'sigma': sigma_list
        }

    def compute(self, gauss_params, x, y):
        # some simple check
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        centers = gauss_params['centers']
        sigma_list = gauss_params['sigma']

        total_gaussian = npo.zeros_like(x)
        for (x0, y0), sigma in zip(centers, sigma_list):
            total_gaussian += npo.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # Normalize total_gaussian so that int(int(total_gaussian dx) dy) = 1
        # very important to match the units in PDE system
        integral, _ = dblquad(
            lambda x_, y_: sum(
                npo.exp(-((x_ - x0)**2 + (y_ - y0)**2) / (2 * sigma**2))
                for (x0, y0), sigma in zip(centers, sigma_list)
            ),
            self.domain_x[0], self.domain_x[1],
            lambda _: self.domain_y[0], lambda _: self.domain_y[1]
        )

        return total_gaussian / integral

