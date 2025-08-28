import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import numpy as npo
import sys
import shutil
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

# external imports
from .config import get_paths
from .data_sampler import GaussianSampler2D
from .map_from_firedrake import map_concentration_fem

def generate_one_test_data(master_rng, problem_config):
    """
    Generate one test dataset using a regular spatial grid
    Returns:
        u_test: Array of shape (Q * Nt, m), input source sampled
        y_test: Array of shape (Q * Nt, 4), (x,y,t,u)
        s_test: Array of shape (Q * Nt,), concentration values at test points.
    """
    # =================================================================
    # Unpack problem configuration
    xmin, xmax = problem_config['space_domain'][0]
    ymin, ymax = problem_config['space_domain'][1]
    tmin, tmax = problem_config['time_domain']
    m = problem_config['m']
    Nt = problem_config['Nt']


    # Sample a Gaussian for the source term
    rngs = [npo.random.default_rng(master_rng.integers(1e9)) for _ in range(3)]
    sampler = GaussianSampler2D(problem_config)
    gauss_params = sampler.sample(rngs[0])

    # use cartesian grid for testing grid
    Nx_test = problem_config.get("test_grid_size", 50)
    x_test = npo.linspace(xmin, xmax, Nx_test)
    y_test = npo.linspace(ymin, ymax, Nx_test)
    xx_test, yy_test = npo.meshgrid(x_test, y_test)
    x_r = xx_test.flatten()
    y_r = yy_test.flatten()
    Q = x_r.shape[0]


    # this is important to be able to get slices
    t_r = npo.linspace(tmin, tmax, Nt)

    # input function repeated for each PDE collocation points
    ux_test = sampler.compute(gauss_params, x_r, y_r)  # (Q,)
    ux_test_flat = npo.tile(ux_test, Nt)                # (Q * Nt,)

    #Solve the convection-diffusion equation using the solver
    c_sol = map_concentration_fem(gauss_params, problem_config, x_r, y_r)

    # Get input function at the sensor points (branch input) and repeat for each space-time combination
    xx = npo.linspace(xmin, xmax, m)
    yy = npo.linspace(ymin, ymax, m)
    XX, YY = npo.meshgrid(xx, yy)                             # (m,m)
    u = sampler.compute(gauss_params, XX, YY).reshape(m, m)   # (m, m)
    u_test = npo.tile(u.flatten(), (Q * Nt, 1))               # (Q * Nt, m * m)


    # super important, pair up each space point with one discrete time
    X_r, T_r = npo.meshgrid(x_r, t_r, indexing='ij')  # (Q, Nt)
    Y_r, T_r = npo.meshgrid(y_r, t_r, indexing='ij')  # (Q, Nt)
    x_flat = X_r.flatten()                            # (Q * Nt,)
    y_flat = Y_r.flatten()                            # (Q * Nt,)
    t_flat = T_r.flatten()                            # (Q * Nt,)


    # Stack the components to form y_test
    y_test = npo.stack([x_flat, y_flat, t_flat, ux_test_flat], axis=1)  # (Q * Nt, 4)

    # Flatten the solution array (UU) to match the space-time points
    s_test = c_sol.flatten()  # [Q * Nt,]

    # Cast to float32, fp16 is not stable for MSE computation
    u_test = u_test.astype(npo.float32)   # (Q * Nt, m * m)
    y_test = y_test.astype(npo.float32)   # (Q * Nt, 4)
    s_test = s_test.astype(npo.float32)   # (Q * Nt,)

    return u_test, y_test, s_test, gauss_params


def generate_test_data(master_rng, problem_config):
    """
    Generate multiple test datasets and save them to a single .npz file.

    Outputs (saved to disk):
    - u_test:  (N_test * Q * Nt, m * m)
    - y_test:  (N_test * Q * Nt, 8)
    - s_test:  (N_test * Q * Nt,)
    - x0:      (N_test, num_gauss)
    - y0:      (N_test, num_gauss)
    - sigma:   (N_test, num_gauss)
    """

    paths = get_paths(problem_config)
    test_data_save_dir = paths['test_data_dir']
    if os.path.exists(test_data_save_dir):
        shutil.rmtree(test_data_save_dir)
    os.makedirs(test_data_save_dir)
    file_path = os.path.join(test_data_save_dir, 'testing_data.npz')

    N_test = problem_config['N_test']
    rngs = [npo.random.default_rng(master_rng.integers(1e9)) for _ in range(N_test)]

    u_test_list = []
    y_test_list = []
    s_test_list = []
    x0_list = []
    y0_list = []
    sigma_list = []

    with trange(N_test, desc="Generating Test Data", leave=True) as progress_bar:
        for i in range(N_test):
            # Generate one test sample
            u_test, y_test, s_test, gauss_params = generate_one_test_data(rngs[i], problem_config)

            # FIX: extract from 'centers', not 'center'
            centers = npo.array(gauss_params['centers'])  # shape: (num_gauss, 2)
            sigmas = npo.array(gauss_params['sigma'])     # shape: (num_gauss,)

            x0_list.append(centers[:, 0])  # (num_gauss,)
            y0_list.append(centers[:, 1])  # (num_gauss,)
            sigma_list.append(sigmas)      # (num_gauss,)

            u_test_list.append(u_test)
            y_test_list.append(y_test)
            s_test_list.append(s_test)

            progress_bar.update(1)

    # Concatenate all test data
    u_test_all = npo.asarray(npo.concatenate(u_test_list, axis=0)).astype(npo.float32)
    y_test_all = npo.asarray(npo.concatenate(y_test_list, axis=0)).astype(npo.float32)
    s_test_all = npo.asarray(npo.concatenate(s_test_list, axis=0)).astype(npo.float32)


    # Save as object arrays (lists of variable-length arrays)
    npo.savez_compressed(
        file_path,
        u_test=u_test_all,
        y_test=y_test_all,
        s_test=s_test_all,
        x0=npo.array(x0_list, dtype=object),
        y0=npo.array(y0_list, dtype=object),
        sigma=npo.array(sigma_list, dtype=object)
    )


def load_one_test_data(problem_config, index_id=0):
    """
    Load a single test sample from saved test data.

    Returns:
    - u_test:       shape (Q * Nt, m * m)
    - y_test:       shape (Q * Nt, d)
    - s_test:       shape (Q * Nt,)
    - gauss_params: dict with keys:
        - 'centers': List of (x, y) tuples, one per Gaussian
        - 'sigma':   List of floats, one per Gaussian
    """
    Nt = problem_config['Nt']
    N_test = problem_config['N_test']

    if index_id >= N_test:
        print(f"index_id:{index_id} cannot exceed {N_test}, switching to index_id={N_test - 1}")
        index_id = N_test - 1

    Q = problem_config['test_grid_size'] ** 2
    num_points_per_sample = Q * Nt


    # data is here
    paths = get_paths(problem_config)
    test_data_save_dir = paths['test_data_dir']

    file_path = os.path.join(test_data_save_dir, 'testing_data.npz')
    data = npo.load(file_path,allow_pickle=True)

    # Load data arrays
    u_test_all = data['u_test']      # shape: (N_test * Q * Nt, m * m)
    y_test_all = data['y_test']      # shape: (N_test * Q * Nt, d)
    s_test_all = data['s_test']      # shape: (N_test * Q * Nt,)


    # Slice to extract one test sample
    start_idx = index_id * num_points_per_sample
    end_idx = (index_id + 1) * num_points_per_sample

    u_test = u_test_all[start_idx:end_idx]  # shape: (Q * Nt, m * m)
    y_test = y_test_all[start_idx:end_idx]  # shape: (Q * Nt, d)
    s_test = s_test_all[start_idx:end_idx]  # shape: (Q * Nt,)

    # Extract Gaussian parameters for this sample
    x0_vec = data['x0'][index_id]        # (num_gauss,)
    y0_vec = data['y0'][index_id]        # (num_gauss,)
    sigma_vec = data['sigma'][index_id]  # (num_gauss,)

    # Optional rounding (e.g., for display/debug)
    x0_rounded = [round(float(x), 2) for x in x0_vec]
    y0_rounded = [round(float(y), 2) for y in y0_vec]
    sigma_rounded = [round(float(s), 3) for s in sigma_vec]

    gauss_params = {
        'centers': list(zip(x0_rounded, y0_rounded)),
        'sigma': sigma_rounded
    }

    return u_test, y_test, s_test, gauss_params
