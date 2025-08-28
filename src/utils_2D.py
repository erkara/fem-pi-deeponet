import shutil
import sys
import os
import json
import numpy as npo
import pyvista as pv
import psutil
import GPUtil
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import jax.numpy as np
from jax.flatten_util import ravel_pytree

# external
from .model_file import PI_DeepONet
from .test_data_generation import generate_one_test_data, load_one_test_data
from .config import get_paths



def save_model(model,problem_config):
    """
    Save the model
    """
    paths = get_paths(problem_config)
    model_path = os.path.join(paths['model_dir'], "pi_deeponet.npy")

    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(model_path, flat_params)
    print(f"model saved at : {model_path}")

def load_model(model, problem_config):
    """
    Assuming the model saved, load the model
    """
    paths = get_paths(problem_config)
    model_path = os.path.join(paths['model_dir'], "pi_deeponet.npy")

    flat_params = np.load(model_path)
    _, unravel_fn = ravel_pytree(model.get_params(model.opt_state))
    params = unravel_fn(flat_params)

    return params

def plot_losses(problem_config,clip_after=10000,l2_threshold=0.3):
    """
    Load training log from file and plot. Plot only the sections
    where the error goes below 0.3 for better visibility.
    """
    # Load logs from JSON file
    paths = get_paths(problem_config)
    log_file = f"{paths['train_results_dir']}/train_log.json"
    with open(log_file, "r") as f:
        logs = json.load(f)

    steps = np.array(logs["step"])
    mask = steps >= clip_after

    # Apply clip mask, nah
    steps = steps[mask]
    loss_total = np.array(logs["loss_total"])[mask]
    loss_ic = np.array(logs["loss_ic"])[mask]
    loss_neumann = np.array(logs["loss_neumann"])[mask]
    loss_res = np.array(logs["loss_res"])[mask]
    test_l2 = np.array(logs["test_l2"])[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ===== Zoomed Test L2 Error =====
    mask_l2 = test_l2 < l2_threshold
    if np.any(mask_l2):
        axes[0].plot(steps[mask_l2], test_l2[mask_l2] * 100, lw=2, label='Test L2 Error (%)')
        axes[0].set_ylabel("Test L2 Error (%)")
        axes[0].set_title(f"Test L2 Error (< {l2_threshold * 100:.0f}%)")
        axes[0].legend()
        axes[0].grid(True)
    else:
        axes[0].set_title("Test L2 Error: No values below threshold")

    # ===== Training Losses (log scale) =====
    axes[1].plot(steps, loss_total, lw=2, label='Total Loss')
    axes[1].plot(steps, loss_ic, lw=2, label='IC Loss')
    axes[1].plot(steps, loss_neumann, lw=2, label='Neumann BC Loss')
    axes[1].plot(steps, loss_res, lw=2, label='PDE Residual Loss')
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss Components")
    axes[1].legend()
    axes[1].grid(True)

    for ax in axes:
        ax.set_xlabel("Training Step")

    plt.tight_layout()
    # save
    fig_path = f"{paths['train_results_dir']}/loss_profiles.png"
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_vtk_file(filename, x_flat, y_flat, values, value_name="s"):
    """
    Use vtk format to output to Paraview, then apply Delunay2D to
    create the mesh on it. This is pretty robust and works well for
    our case
    """
    # Stack x, y, and z (0 for 2D) into a points array, no support for fp16
    points = npo.column_stack((x_flat, y_flat, npo.zeros_like(x_flat))).astype(npo.float32)
    values = values.astype(npo.float32)

    # save grid to vtk
    grid = pv.PolyData(points)
    grid.point_data[value_name] = values
    grid.save(filename)

def save_vtk_comparison(y_test, s_test, s_pred, problem_config):
    """
    Save the time slices of FEM and DeepONet results to VTK for Paraview visualization.
    """

    paths = get_paths(problem_config)
    vtk_dir = paths['vtk_dir']
    if os.path.exists(vtk_dir):
        shutil.rmtree(vtk_dir)
    os.makedirs(vtk_dir)

    Nt = problem_config['Nt']
    Q = s_test.shape[0] // Nt

    x_flat = npo.asarray(y_test[:, 0].reshape(Q, Nt)[:, -1])
    y_flat = npo.asarray(y_test[:, 1].reshape(Q, Nt)[:, -1])
    s_test = s_test.reshape(Q, Nt)
    s_pred = s_pred.reshape(Q, Nt)

    for t_idx in range(Nt):
        test_outfile = os.path.join(vtk_dir, f"fem_test_data_{t_idx:03d}.vtk")
        pred_outfile = os.path.join(vtk_dir, f"model_predictions_{t_idx:03d}.vtk")

        save_vtk_file(test_outfile, x_flat, y_flat, s_test[:, t_idx], value_name="FEM")
        save_vtk_file(pred_outfile, x_flat, y_flat, s_pred[:, t_idx], value_name="PI-DeepONet")

def _slice_and_interpolate(x_flat, y_flat, s_field, target_y, tol=1e-2, Nt=None):
    """
    Extract 1D slice at y = target_y and return interpolated profile (x_interp, s_interp).
    """
    Q = len(x_flat)
    y_diffs = npo.abs(y_flat - target_y)
    idx = npo.where(y_diffs < tol)[0]

    if len(idx) == 0:
        closest = npo.argmin(y_diffs)
        target_y = y_flat[closest]
        idx = npo.where(npo.abs(y_flat - target_y) < tol)[0]

    x_vals = x_flat[idx]
    s_vals = s_field[idx]

    sort_idx = npo.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    s_sorted = s_vals[sort_idx]
    x_unique, idx_unique = npo.unique(x_sorted, return_index=True)

    s_unique = s_sorted[idx_unique]
    x_interp = npo.linspace(x_unique.min(), x_unique.max(), 100)
    s_interp = interp1d(x_unique, s_unique, kind='linear', fill_value="extrapolate")(x_interp)

    return x_interp, s_interp, target_y

def plot_centerline_profiles(y_test, s_test, s_pred, gauss_params, problem_config):

    T = problem_config['time_domain'][1]
    Nt = problem_config['Nt']
    Q = s_test.shape[0] // Nt

    s_test = s_test.reshape(Q, Nt)
    s_pred = s_pred.reshape(Q, Nt)
    x_flat = y_test[:, 0].reshape(Q, Nt)[:, -1]
    y_flat = y_test[:, 1].reshape(Q, Nt)[:, -1]


    # ==per-gaussian  ===
    centers = gauss_params['centers']
    sigmas = gauss_params['sigma']
    num_gauss = len(centers)
    fig, axes = plt.subplots(1, num_gauss, figsize=(5 * num_gauss, 5), squeeze=False)

    # === Plot each individual Gaussian slice ===
    for i, (_, y0) in enumerate(centers):
        x_i, test_i, y_actual = _slice_and_interpolate(x_flat, y_flat, s_test[:, -1], y0)
        _, pred_i, _ = _slice_and_interpolate(x_flat, y_flat, s_pred[:, -1], y0)

        ax = axes[0, i]
        ax.plot(x_i, test_i, 'b-', label='FEM')
        ax.plot(x_i, pred_i, 'r--', label='PI-DeepONet')
        x_center = float(round(centers[i][0], 2))
        y_center = float(round(centers[i][1], 2))
        ax.set_title(fr'Source: {i + 1}: $x_0$ = ({x_center:.2f}, {y_center:.2f}), $\sigma$={sigmas[i]:0.2f}')
        ax.set_xlabel('$x$')
        ax.legend()

    plt.tight_layout()
    paths = get_paths(problem_config)
    plt.savefig(f"{paths['train_results_dir']}/centerline_profiles.png")
    plt.close(fig)

def test_model(testing_rng, problem_config, model_config):
    """
    test the trained model for N_test independent data
    """
    model = PI_DeepONet(model_config, problem_config)
    params = load_model(model, problem_config)
    N_test = problem_config['N_test']
    Nt = problem_config['Nt']
    t_min, t_max = problem_config['time_domain']

    total_rel_l2 = 0
    timewise_l2_numerators = npo.zeros(Nt)
    timewise_l2_denominators = npo.zeros(Nt)

    rngs = [npo.random.default_rng(testing_rng.integers(1e9)) for _ in range(N_test)]

    for i in tqdm(range(N_test), desc="testing"):
        u_test, y_test, s_test, gauss_params = generate_one_test_data(rngs[i], problem_config)
        s_pred = model.predict_s(params, u_test, y_test)

        Q = s_test.shape[0] // Nt
        s_test = s_test.reshape(Q, Nt)
        s_pred = s_pred.reshape(Q, Nt)

        # Full error for this sample
        err_full = npo.linalg.norm(s_test - s_pred) / npo.linalg.norm(s_test)
        total_rel_l2 += err_full

        for t in range(Nt):
            diff = s_test[:, t] - s_pred[:, t]
            timewise_l2_numerators[t] += npo.linalg.norm(diff)**2
            timewise_l2_denominators[t] += npo.linalg.norm(s_test[:, t])**2

    # Final mean errors
    mean_relative_l2 = total_rel_l2 / N_test
    per_time_rel_l2_array = npo.sqrt(timewise_l2_numerators / timewise_l2_denominators)

    # Map to time -> rel_l2 dict
    t_grid = npo.linspace(t_min, t_max, Nt)
    per_time_rel_l2_by_time = {
        float(round(t, 0)): float(round(100 * err, 6))
        for t, err in zip(t_grid[1:], per_time_rel_l2_array[1:])   # first is IC, not reliable.
    }

    # save for the record
    paths = get_paths(problem_config)
    log_file = f"{paths['train_results_dir']}/relative_l2_testing_logs.json"
    results = {
        'mean_relative_l2': round(float(100 * mean_relative_l2),6),
        'per_time_relative_l2': per_time_rel_l2_by_time
    }
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)

    return mean_relative_l2, per_time_rel_l2_by_time

def compare_speed(testing_rng, problem_config, model_config):
    """
    This is to compare FEM and PI-DeepONet run-time performance during inference.
    """
    from .map_from_firedrake import solve_cd_for_speed_test
    import time

    model = PI_DeepONet(model_config, problem_config)
    params = load_model(model, problem_config)

    N_test = problem_config['N_test']
    final_sim_time = problem_config['time_domain'][1]
    rngs = [npo.random.default_rng(testing_rng.integers(1e9)) for _ in range(N_test)]

    fem_time_total = 0
    model_time_total = 0

    fem_times = []
    model_times = []

    for i in tqdm(range(N_test), desc="testing"):
        u_test, y_test, s_test, gauss_params = generate_one_test_data(rngs[i], problem_config)

        # FEM timing (removed all I/Os for fair comparison)
        simulation_time_transport = solve_cd_for_speed_test(gauss_params, problem_config)
        fem_times.append(round(simulation_time_transport, 6))
        fem_time_total += simulation_time_transport

        # Model timing
        start_time = time.time()
        _ = model.predict_s(params, u_test, y_test)
        simulation_time_model = time.time() - start_time
        model_times.append(round(simulation_time_model, 6))
        model_time_total += simulation_time_model

    avg_time_fem = round(fem_time_total / N_test, 6)
    avg_time_model = round(model_time_total / N_test, 6)
    speedup_ratio = round(avg_time_fem / avg_time_model, 3)

    results = {
        "avg_time_fem": avg_time_fem,
        "avg_time_model": avg_time_model,
        "avg_time_fem/avg_time_model": speedup_ratio,
        "final_sim_time": final_sim_time,
        "fem_times": fem_times,
        "model_times": model_times
    }

    paths = get_paths(problem_config)
    log_file = f"{paths['train_results_dir']}/speed_comparison.json"

    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)

    return avg_time_fem, avg_time_model




## misc utils
def check_for_nans_numpy(arr, name):
    has_nans = npo.isnan(arr).any()
    if has_nans:
        print(f"\nWarning: NaN detected in {name}")
    return has_nans

def get_memory_usage():
    # CPU memory usage
    process = psutil.Process()
    process_memory_gb = process.memory_info().rss / (1024 ** 3)
    system_memory = psutil.virtual_memory()
    total_memory_gb = system_memory.total / (1024 ** 3)
    available_memory_gb = system_memory.available / (1024 ** 3)
    memory_percent = system_memory.percent

    # GPU memory usage
    gpu = GPUtil.getGPUs()[0]  # Only get the first GPU
    total_gpu_memory_gb = gpu.memoryTotal / 1024
    used_gpu_memory_gb = gpu.memoryUsed / 1024
    free_gpu_memory_gb = gpu.memoryFree / 1024
    gpu_memory_percent = gpu.memoryUtil * 100

    # keep track of this, it saves me all the time
    cpu_info = [
        f"Process Memory: {process_memory_gb:.2f} GB",
        f"Total Memory: {total_memory_gb:.2f} GB",
        f"Available Memory: {available_memory_gb:.2f} GB",
        f"Memory Usage: {memory_percent}%"
    ]

    gpu_info = [
        f"Total GPU Memory: {total_gpu_memory_gb:.2f} GB",
        f"Used GPU Memory: {used_gpu_memory_gb:.2f} GB",
        f"Free GPU Memory: {free_gpu_memory_gb:.2f} GB",
        f"GPU Memory Usage: {gpu_memory_percent:.2f}%"
    ]
    print("{:<30} | {:<30}".format("CPU Info", "GPU Info"))
    print("-" * 63)
    for cpu, gpu in zip(cpu_info, gpu_info):
        print("{:<30} | {:<30}".format(cpu, gpu))

