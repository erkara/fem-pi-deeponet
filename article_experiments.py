import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import copy
import time
import json
import yaml
from tqdm.auto import tqdm
import jax.numpy as np
import numpy as npo

from jax import clear_caches
from src.train_data_generation import load_training_data_from_chunks, generate_training_data_with_chunks
from src.test_data_generation import load_one_test_data, generate_one_test_data
from src.utils_2D import (load_model, save_vtk_comparison, plot_centerline_profiles,
                          plot_losses, test_model, test_model,
                          compare_speed, get_memory_usage, get_paths)
from src.model_file import PI_DeepONet, DataGenerator


# ====== Base Configs ======
BASE_PROBLEM_CONFIG = {
    'DIM': 3,
    'space_domain': ((0, 10), (0, 10)),
    'time_domain': (0, 500),
    'N': 20,
    'm': 30,
    'P_bcs': 100,
    'P_ics': 5,
    'n_r': 20,
    'n_theta': 20,
    'n_random': 300,
    'center_range': ((3, 7), (3, 7)),
    'percentage_data_to_load': 1.,
    'precision': 'float16',
    'test_grid_size': 80,
    'Nt': 10,
    'N_test': 30,
    'TRAIN_MODE': True,
    'log_int': 1000,
    'save_pde_simulation': False,
    'K': 1e-9,
    'mu_w': 9e-4,
    'alpha': 1e-2,
    'D': 4e-6,
    'beta1': 5 / 60,
    'beta2': 0.25 * (5 / 60),
    'SUPG': False,
}

BASE_MODEL_CONFIG = {
    'branch_layers': [BASE_PROBLEM_CONFIG['m']**2, 128, 128, 128, 128],
    'trunk_layers': [3, 128, 128, 128, 128],
    'batch_size': 200,
    'num_epoch': 2000,
    'initial_lr': 0.001,
    'decay_steps': 5000,
    'decay_rate': 0.95
}


def run_experiment(exp_name, base_problem, base_model, time_domain):
    print(f"\n======================== Running {exp_name} =====================")
    #get_memory_usage()

    problem_config = copy.deepcopy(BASE_PROBLEM_CONFIG)
    model_config = copy.deepcopy(BASE_MODEL_CONFIG)

    # overrides
    problem_config.update(base_problem)
    model_config.update(base_model)
    problem_config["time_domain"] = time_domain
    problem_config["model_name"] = exp_name

    # sanity check
    print(f" Final Time: {time_domain[1]}")
    print(f" Branch Layers: {model_config['branch_layers']}")
    print(f" Trunk Layers:  {model_config['trunk_layers']}")

    # this fully governs randomness
    master_rng = npo.random.default_rng(seed=1234)
    testing_rng = npo.random.default_rng(seed=789)

    # Construct model
    model = PI_DeepONet(model_config, problem_config)

    # Generate test data
    u_test, y_test, s_test, gauss_params = generate_one_test_data(testing_rng, problem_config)

    if problem_config['TRAIN_MODE']:
        ##===============Generate and Load Train Data==============
        timings = {}
        start_time = time.time()
        generate_training_data_with_chunks(master_rng, problem_config)
        timings["train_data_generation_time"] = round(time.time() - start_time, 6)

        paths = get_paths(problem_config)
        log_file = f"{paths['train_results_dir']}/data_generation_time.json"


        # Load training data
        (u_ic, y_ics, s_ics,
         u_bc, y_bcs,
         u_res, y_res, s_res) = load_training_data_from_chunks(master_rng, problem_config)

        ic_dataset = DataGenerator(u_ic, y_ics, s_ics, model_config['batch_size'])
        bc_dataset = DataGenerator(u_bc, y_bcs, np.zeros_like(y_bcs[:, :1]), model_config['batch_size'])
        res_dataset = DataGenerator(u_res, y_res, s_res, model_config['batch_size'])

        get_memory_usage()

        #===================Train the model========================
        start_time = time.time()
        model.train(ic_dataset, bc_dataset, res_dataset,
                    test_data=(u_test, y_test, s_test),
                    eval_full_test_steps=[])
        timings["training_time"] = round(time.time() - start_time, 6)
        with open(log_file, "w") as f:
            json.dump(timings, f, indent=2)

        # plot residual/bc/ic/total losses
        plot_losses(problem_config)
        clear_caches()

        # get predictions
        params = load_model(model, problem_config)
        s_pred = model.predict_s(params, u_test, y_test)

        # save the Nt time steps to vtk to view in Paraview
        save_vtk_comparison(y_test, s_test, s_pred, problem_config)

        # save mean + per-gaussian y_center profiles
        plot_centerline_profiles(y_test, s_test, s_pred, gauss_params, problem_config)

        # report error metrics over N_test cases
        print("=================================================")
        test_results = test_model(testing_rng, problem_config, model_config)
        print(f"mean relative error over {problem_config['N_test']} testing: {test_results['mean_relative_l2']:0.2f} %")
        print(f"final-time stats: {test_results['final_time_stats']}")
        print(f"time wise relative errors: {test_results['per_time_relative_l2']}")
    else:
        # do this if the model is saved/ready using TRAIN_MODE switch up above
        params = load_model(model, problem_config)
        s_pred = model.predict_s(params, u_test, y_test)
        save_vtk_comparison(y_test, s_test, s_pred, problem_config)
        plot_centerline_profiles(y_test, s_test, s_pred, gauss_params, problem_config)


        # mean_rel, per_time = test_model(testing_rng, problem_config, model_config)
        # print(f"Mean Relative L2 Error = {100 * mean_rel:.2f}%")
        # print(f"Per-time L2 errors: {per_time}")


def main():
    RUN_SINGLE = True         # run a single experiment or all
    SINGLE_EXPERIMENT_NAME = "single_source_T50"

    # Load YAML config (families + variants)
    with open("src/experiment_configs.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    # single_source, double_source_random, triple_source, triple_source_fixed_center
    families = config_data["experiment_families"]

    # holding varients
    variants = config_data["experiment_variants"]

    if RUN_SINGLE:
        #e.g {"family": "single_source","time_domain": [0, 500]}
        v = variants[SINGLE_EXPERIMENT_NAME]

        # shared config for that family
        #e.g "problem_config": { "gauss_mode": ..., "sigma_range": ... },"model_config": { "num_epoch": ..., ... } }
        fam = families[v["family"]]
        # Run the experiment with full configs
        run_experiment(SINGLE_EXPERIMENT_NAME, fam["problem_config"], fam["model_config"], v["time_domain"])
    else:
        # Loop through all defined variants and run each
        for name, v in variants.items():
            fam = families[v["family"]]
            run_experiment(name, fam["problem_config"], fam["model_config"], v["time_domain"])
            clear_caches()

if __name__ == '__main__':
    main()

