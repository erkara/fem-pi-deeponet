import os
import sys
import csv
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%m-%d-%y-%H:%M:%S")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import shutil
import numpy as npo
from tqdm import trange
from scipy.integrate import dblquad
# external imports
from .map_from_firedrake import mapped_velocity_fem
from .data_sampler import GaussianSampler2D

from .utils_2D import check_for_nans_numpy
from .config import get_paths
save_interval = 5


def generate_boundary_points(master_rng, problem_config):
    """
    Generate boundary and initial condition points with independent random generators for each part.
    Combining them is a pretty common practice
    """
    P_bcs = problem_config['P_bcs']
    P_ics = problem_config['P_ics']  # Use P_ics from problem_config
    _, xmax = problem_config['space_domain'][0]
    _, ymax = problem_config['space_domain'][1]
    tmin, tmax = problem_config['time_domain']

    # Adjust P_bcs to be divisible by 4 if needed, fine since we have rectangular domains
    if P_bcs % 4 != 0:
        P_bcs = int(4 * round(P_bcs / 4))

    P_per_boundary = P_bcs // 4

    # Generate independent RNG streams from the master RNG
    rngs = [npo.random.default_rng(master_rng.integers(1e9)) for _ in range(7)]

    # Generate boundary points for x = 0 and x = xmax (varying y)
    x_bc1 = npo.zeros((P_per_boundary, 1))                               # x = 0 boundary
    y_bc1 = rngs[0].uniform(low=0, high=ymax, size=(P_per_boundary, 1))  # Varying y on x = 0
    x_bc3 = xmax * npo.ones((P_per_boundary, 1))                         # x = xmax boundary
    y_bc3 = rngs[1].uniform(low=0, high=ymax, size=(P_per_boundary, 1))  # Varying y on x = xmax

    # Generate boundary points for y = 0 and y = ymax (varying x)
    y_bc2 = npo.zeros((P_per_boundary, 1))                               # y = 0 boundary
    x_bc2 = rngs[2].uniform(low=0, high=xmax, size=(P_per_boundary, 1))  # Varying x on y = 0
    y_bc4 = ymax * npo.ones((P_per_boundary, 1))                         # y = ymax boundary
    x_bc4 = rngs[3].uniform(low=0, high=xmax, size=(P_per_boundary, 1))  # Varying x on y = ymax

    # Combine all boundary points
    x_bcs = npo.vstack([x_bc1, x_bc3, x_bc2, x_bc4])  # Combine x boundary points
    y_bcs = npo.vstack([y_bc1, y_bc3, y_bc2, y_bc4])  # Combine y boundary points

    # Generate random time points for boundary conditions
    t_bcs = rngs[4].uniform(low=tmin, high=tmax, size=(P_bcs, 1))  # Random time points

    # Combine boundary points
    y_train_boundary_space = npo.hstack([x_bcs, y_bcs, t_bcs])  # Combine x, y, and t for boundary points

    # Initial condition points (random points within the domain, t = 0)
    x_ic = rngs[5].uniform(low=0, high=xmax, size=(P_ics, 1))
    y_ic = rngs[6].uniform(low=0, high=ymax, size=(P_ics, 1))
    t_ic = npo.zeros((P_ics, 1))  # t = 0 for initial conditions

    # Combine initial condition points
    y_train_initial = npo.hstack([x_ic, y_ic, t_ic])   # Combine x, y, and t for initial conditions
    s_train_ic = npo.zeros((P_ics, 1))                 # Solution values at initial points (zeros)


    return y_train_boundary_space, y_train_initial, s_train_ic




def generate_gauss_grid(master_rng, gauss_params, problem_config,
                        n_r, n_theta, n_random):
    """
    Generate a grid consisting of:
    - "n_r * n_theta" structured polar points around each Gaussian center within std_factor * sigma
    - "n_random" uniform random points in the domain shared for all Gaussians
    """
    xmin, xmax = problem_config['space_domain'][0]
    ymin, ymax = problem_config['space_domain'][1]
    std_factor = problem_config['std_factor']
    centers = gauss_params['centers']
    sigmas = gauss_params['sigma']
    rngs = [npo.random.default_rng(master_rng.integers(1e9)) for _ in range(3)]

    x_polar_list = []
    y_polar_list = []

    for (x0, y0), sigma in zip(centers, sigmas):
        # avoid falling out of domain if std_factor is big
        r_max = min(std_factor * sigma, x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
        if r_max < std_factor * sigma:
            print(f"Clipped radius at center ({x0:.2f}, {y0:.2f}) to r_max = {r_max:.2f}")
        r_min = r_max / (n_r * 100)
        r = npo.linspace(r_min, r_max, n_r)
        theta = npo.linspace(0, 2 * npo.pi, n_theta, endpoint=False)
        r_grid, theta_grid = npo.meshgrid(r, theta)
        x_polar = x0 + r_grid * npo.cos(theta_grid)
        y_polar = y0 + r_grid * npo.sin(theta_grid)
        x_polar_list.append(x_polar.flatten())
        y_polar_list.append(y_polar.flatten())

    # Combine structured polar points from all Gaussians
    x_polar_all = npo.concatenate(x_polar_list)
    y_polar_all = npo.concatenate(y_polar_list)

    # Random points shared across all Gaussians
    x_random = rngs[1].uniform(low=xmin, high=xmax, size=n_random)
    y_random = rngs[2].uniform(low=ymin, high=ymax, size=n_random)

    # Combine polar and random points
    x_r = npo.concatenate([x_polar_all, x_random])
    y_r = npo.concatenate([y_polar_all, y_random])

    return x_r, y_r



def generate_one_training_data(master_rng, problem_config):
    """
    Generate one training sample for the 2D convection-diffusion problem, consisting of:
    - Initial condition points (supervised: c = 0).
    - Boundary points (used for Neumann condition, no target).
    - PDE residual points (used for PDE residual loss).

    Be obsessed with the shapes
        u_ic:        (P_ics, m*m)   input function repeated for IC points
        y_ics:       (P_ics, 3)     IC collocation points
        s_ics:       (P_ics, 1)     target values at IC points (all zero)

        u_bc:        (P_bcs, m*m)   input function repeated for boundary points
        y_bcs:       (P_bcs, 3)     boundary collocation points (no target)

        u_r_train:   (Q, m*m)       input function repeated for residual points
        y_r_train:   (Q, 8)         PDE residual features: x, y, t, u, v1, v2, dv1dx, dv2dy
        s_r_train:   (Q, 1)         zeros for residual loss
    """
    #=====================================================
    # problem configuration
    xmin, xmax = problem_config['space_domain'][0]
    ymin, ymax = problem_config['space_domain'][1]
    tmin, tmax = problem_config['time_domain']
    n_r, n_theta, n_random = problem_config['n_r'], problem_config['n_theta'], problem_config['n_random']

    m = problem_config['m']                                          # sample on m * m grid for branch
    precision = problem_config.get('precision', 'float16')           # use to save fp16/fp32
    dtype = npo.float16 if precision == 'float16' else npo.float32

    rngs = [npo.random.default_rng(master_rng.integers(1e9)) for _ in range(10)]

    # Sample a Gaussian source function at m * m sensor points
    sampler = GaussianSampler2D(problem_config)
    gauss_params = sampler.sample(rngs[0])
    xx = npo.linspace(xmin, xmax, m)
    yy = npo.linspace(ymin, ymax, m)
    XX, YY = npo.meshgrid(xx, yy)
    u = sampler.compute(gauss_params, XX, YY).reshape(m, m)

    # Generate BC and IC points independently
    # IC has targets (s = 0), BC is only used in physics residual loss (Neumann)
    y_bcs, y_ics, s_ics = generate_boundary_points(rngs[1], problem_config)

    # For each BC/IC or residual point, we attach the SAME u
    # per point of interest, otherwise the shapes do not match. Recall that
    # u is the global context while each point is local.
    u_bc = npo.tile(u.flatten(), (y_bcs.shape[0], 1))   # (P_bcs, m * m)
    u_ic = npo.tile(u.flatten(), (y_ics.shape[0], 1))   # (P_ics, m * m)

    # PDE collocation points
    x_r, y_r = generate_gauss_grid(rngs[2], gauss_params, problem_config, n_r, n_theta, n_random)
    Q = x_r.shape[0]

    # temporal time sampling
    t_r = rngs[3].uniform(low=tmin, high=tmax, size=Q)

    # We need input function values at PDE collocation points for PDE residuals
    u_r = sampler.compute(gauss_params, x_r, y_r)  # (Q,)

    # Get the velocity field and its divergence from the FEM solver
    v1_r, v2_r, dv1dx_r, dv2dy_r = mapped_velocity_fem(problem_config, gauss_params, x_r, y_r)  # (Q,)

    # Repeat input function profile for each PDE collocation point
    u_r_train = npo.tile(u.flatten(), (Q, 1))  # (Q, m * m)

    # Pack residual coordinates and auxiliary fields: (x_r, y_r, t_r, u_r, v1_r, v2_r, dv1/dx, dv2/dy)
    # TODO: pack this for better readibility
    y_r_train = npo.vstack([x_r, y_r, t_r, u_r, v1_r, v2_r, dv1dx_r, dv2dy_r]).T  # (Q, 8)

    # PDE residual targets are always zero
    s_r_train = npo.zeros((Q, 1))  # Zero array for PDE residual (Q, 1)

    # Convert all arrays to desired precision
    # typical data size ~ 10GB, use fp16, performance seems to be fine.
    u_ic = u_ic.astype(dtype)
    y_ics = y_ics.astype(dtype)
    s_ics = s_ics.astype(dtype)
    u_bc = u_bc.astype(dtype)
    y_bcs = y_bcs.astype(dtype)
    u_r_train = u_r_train.astype(dtype)
    y_r_train = y_r_train.astype(dtype)
    s_r_train = s_r_train.astype(dtype)

    ##keep this here no matter what!##
    def check_and_exit_if_nan_detected():
        arrays_to_check = {
            "u_ic": u_ic,
            "y_ics": y_ics,
            "s_ics": s_ics,
            "u_bc": u_bc,
            "y_bcs": y_bcs,
            "u_r_train": u_r_train,
            "y_r_train": y_r_train,
            "s_r_train": s_r_train
        }

        for name, array in arrays_to_check.items():
            if check_for_nans_numpy(array, name):
                print(f"NaNs detected during data generation in {name}, exiting!")
                sys.exit('')
    check_and_exit_if_nan_detected()
    ##================================================

    return u_ic, y_ics, s_ics, u_bc, y_bcs, u_r_train, y_r_train, s_r_train




###============

def calculate_data_size(master_rng, problem_config):
    """
    Helper function to calculate the size of a single and total training data batch,
    based on current problem config. Useful to check storage requirements before generation.
    """
    N = problem_config['N']  # Number of training samples

    # Get one training sample
    (u_ic, y_ics, s_ics,
     u_bc, y_bcs,
     u_res, y_res, s_res) = generate_one_training_data(master_rng, problem_config)

    single_train_byte = (
        u_ic.nbytes + y_ics.nbytes + s_ics.nbytes +
        u_bc.nbytes + y_bcs.nbytes +
        u_res.nbytes + y_res.nbytes + s_res.nbytes
    )

    # report the results
    single_train_size = single_train_byte / (1024 ** 2)
    single_saved_chunk = (save_interval * single_train_byte) / (1024 ** 2)  # MB per chunk
    total_train_size = (N * single_train_byte) / (1024 ** 3)

    print(f"single training data: {single_train_size:0.1f} MB\n"
          f"single saved chunk: {single_saved_chunk:0.1f} MB\n"
          f"total training data: {total_train_size:0.2f} GB")


def save_training_data(u_ic, y_ics, s_ics,
                       u_bc, y_bcs,
                       u_res, y_res, s_res,
                       meta, file_path):
    """
    Save training data chunk consisting of:
    - Initial condition data (supervised)
    - Boundary condition data (unsupervised, Neumann)
    - PDE residual collocation data

    Data is compressed to save disk space, super useful if you ssh stuff to cloud
    """

    npo.savez_compressed(file_path,
                         u_ic=u_ic, y_ics=y_ics, s_ics=s_ics,
                         u_bc=u_bc, y_bcs=y_bcs,
                         u_res=u_res, y_res=y_res, s_res=s_res,
                         **meta)


def generate_training_data_with_chunks(master_rng, problem_config):
    """
    Generate and save training data in chunks for efficiency and memory management.
    Idea is to split the training data into smaller chunks to

    Each training sample includes:
    - Initial condition points (supervised)
    - Boundary condition points (Neumann, unsupervised)
    - PDE residual points (unsupervised)

    We save `save_interval` samples per chunk to reduce the overhead of I/O operations.

    Keep an eye on shapes
    - u_ic:        (save_interval * P_ics, m * m)
    - y_ics:       (save_interval * P_ics, 3)
    - s_ics:       (save_interval * P_ics, 1)
    - u_bc:        (save_interval * P_bcs, m * m)
    - y_bcs:       (save_interval * P_bcs, 3)
    - u_res:       (save_interval * Q, m * m)
    - y_res:       (save_interval * Q, 8)
    - s_res:       (save_interval * Q, 1)
    """
    # Remove and recreate the save_path folder to store the data
    paths = get_paths(problem_config)
    train_data_dir = paths['train_data_dir']
    if os.path.exists(train_data_dir):
        shutil.rmtree(train_data_dir)
    os.makedirs(train_data_dir)

    # Get config
    N = problem_config['N']
    rngs = [npo.random.default_rng(master_rng.integers(1e9)) for _ in range(N)]

    # Temporary storage lists
    u_ic_list, y_ics_list, s_ics_list = [], [], []
    u_bc_list, y_bcs_list = [], []
    u_res_list, y_res_list, s_res_list = [], [], []

    total_current_size = 0

    with trange(N, desc="Generating Training Data", leave=True) as progress_bar:
        for i in range(N):
            # Generate one training sample
            (u_ic, y_ics, s_ics,
             u_bc, y_bcs,
             u_res, y_res, s_res) = generate_one_training_data(rngs[i], problem_config)

            # Determine residual batch size
            Q = u_res.shape[0]

            # Estimate size from first batch
            if i == 0:
                first_batch_size = (
                        u_ic.nbytes + y_ics.nbytes + s_ics.nbytes +
                        u_bc.nbytes + y_bcs.nbytes +
                        u_res.nbytes + y_res.nbytes + s_res.nbytes
                )
                size_first_chunk = first_batch_size / (1024 ** 2)  # MB

            total_current_size += size_first_chunk

            # Append to chunk buffer
            u_ic_list.append(u_ic)
            y_ics_list.append(y_ics)
            s_ics_list.append(s_ics)

            u_bc_list.append(u_bc)
            y_bcs_list.append(y_bcs)

            u_res_list.append(u_res)
            y_res_list.append(y_res)
            s_res_list.append(s_res)

            # Save chunk
            if (i + 1) % save_interval == 0 or i == N - 1:
                file_path = f"{train_data_dir}/data_part_{i // save_interval}.npz"

                meta = {
                    'Q': Q
                }

                save_training_data(
                    npo.vstack(u_ic_list),
                    npo.vstack(y_ics_list),
                    npo.vstack(s_ics_list),
                    npo.vstack(u_bc_list),
                    npo.vstack(y_bcs_list),
                    npo.vstack(u_res_list),
                    npo.vstack(y_res_list),
                    npo.vstack(s_res_list),
                    meta,
                    file_path
                )

                # Clear buffers
                u_ic_list, y_ics_list, s_ics_list = [], [], []
                u_bc_list, y_bcs_list = [], []
                u_res_list, y_res_list, s_res_list = [], [], []


                progress_bar.set_postfix({
                    'chunk ID': (i + 1) // save_interval,
                    'location': file_path,
                    'data_size': f'{total_current_size:0.1f} MB'
                })

            progress_bar.update(1)

    # check the sizes
    calculate_data_size(master_rng, problem_config)


def load_training_data_from_chunks(master_rng, problem_config, file_prefix='data_part_'):
    """
    Here, we are loading data chunk-by-chunk with the option to load

    Returns:
        u_ic:   (N_ics_total, m*m)
        y_ics:  (N_ics_total, 3)
        s_ics:  (N_ics_total, 1)

        u_bc:   (N_bcs_total, m*m)
        y_bcs:  (N_bcs_total, 3)

        u_res:  (N_res_total, m*m)
        y_res:  (N_res_total, 8)
        s_res:  (N_res_total, 1)
    """
    #train data should be here.
    paths = get_paths(problem_config)
    train_data_dir = paths['train_data_dir']

    # Lists to collect loaded arrays
    u_ic_list, y_ics_list, s_ics_list = [], [], []
    u_bc_list, y_bcs_list = [], []
    u_res_list, y_res_list, s_res_list = [], [], []

    # unpac some params
    N = problem_config['N']
    m = problem_config['m']
    P_ics = problem_config['P_ics']
    P_bcs = problem_config['P_bcs']

    # Determine which chunks to load
    total_chunks = int(N/ save_interval)
    num_chunks_to_load = int(problem_config['percentage_data_to_load'] * total_chunks)
    selected_chunk_IDs = master_rng.choice(total_chunks, size=num_chunks_to_load, replace=False)

    meta = None
    for idx, i in enumerate(selected_chunk_IDs):
        data = npo.load(f"{train_data_dir}/{file_prefix}{i}.npz", mmap_mode='r')
        if idx == 0:
            meta = {key: data[key] for key in ['Q']}

        #  hate this
        if i == selected_chunk_IDs[-1] and N % save_interval != 0:
            current_save_interval = N % save_interval
        else:
            current_save_interval = save_interval

        # first grab the right sizes
        Q = int(meta['Q'])
        ic_size = current_save_interval * P_ics
        bc_size = current_save_interval * P_bcs
        res_size = current_save_interval * Q

        # IC/BC conditions
        u_ic_list.append(data['u_ic'][:ic_size])
        y_ics_list.append(data['y_ics'][:ic_size])
        s_ics_list.append(data['s_ics'][:ic_size])
        u_bc_list.append(data['u_bc'][:bc_size])
        y_bcs_list.append(data['y_bcs'][:bc_size])

        # Residual collocation points
        u_res_list.append(data['u_res'][:res_size])
        y_res_list.append(data['y_res'][:res_size])
        s_res_list.append(data['s_res'][:res_size])

    # Stack them all
    u_ic = npo.vstack(u_ic_list).reshape(-1, m * m)
    y_ics = npo.vstack(y_ics_list).reshape(-1, 3)
    s_ics = npo.vstack(s_ics_list).reshape(-1, 1)

    u_bc = npo.vstack(u_bc_list).reshape(-1, m * m)
    y_bcs = npo.vstack(y_bcs_list).reshape(-1, 3)

    u_res = npo.vstack(u_res_list).reshape(-1, m * m)
    y_res = npo.vstack(y_res_list).reshape(-1, 8)
    s_res = npo.vstack(s_res_list).reshape(-1, 1)

    return u_ic, y_ics, s_ics, u_bc, y_bcs, u_res, y_res, s_res




