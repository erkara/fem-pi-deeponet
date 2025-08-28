# TODO: switch to argparse at some point
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
from jax import clear_caches
import jax.numpy as np
import numpy as npo

#====
from src.train_data_generation import load_training_data_from_chunks, generate_training_data_with_chunks
from src.test_data_generation import load_one_test_data, generate_one_test_data
from src.utils_2D import (load_model, save_vtk_comparison,plot_centerline_profiles,
                      plot_losses, test_model)
from src.utils_2D import get_memory_usage
from src.model_file import PI_DeepONet, DataGenerator


def main():
    print(f"Initial Memory Load:")
    get_memory_usage()
    # control randomness in data generation
    master_rng = npo.random.default_rng(seed=1231)
    testing_rng = npo.random.default_rng(seed=3456)
    problem_config = {
         # utils
        'TRAIN_MODE': True,            # Train/Test model
        'log_int': 1000,               # Logging frequency of the results
        'model_name': 'test_model',    # model named to be saved(parent folder)
        # problem params
        'DIM': 3,
        'space_domain': ((0, 10),
                         (0, 10)),
        'time_domain': (0, 500),       # time domain
        'N': 2000,                     # number of input functions sampled
        'm': 30,                       # number of input function sensors--> m*m
        'P_bcs': 100,                  # number of BC collocation points
        'P_ics': 5,                    # number of IC collocation points
        'n_r': 30,                     # number of radial PDE collocation
        'n_theta': 30,                 # number of angular PDE collocation
        'n_random': 300,               # number of random PDE collocation points
        'gauss_mode': {
            'type': 'fixed',             # 'fixed' or 'variable'
            'value': 1                   # number of sources
        },
        'center_range': ((4.75, 4.75),   # center range of Gaussian
                         (5.53, 5.53)),
        'sigma_range': (0.25,0.60),      # range of Gaussian width

        'fixed_center': False,           # fixed centers or sample
        'std_factor': 3,                 # radius factor around source, std_factor * sigma
        'percentage_data_to_load': 1.,   # keep for testing
        'precision': 'float16',          # precision of train data to save-up space
        'test_grid_size': 80,            # testing grid
        'Nt': 10,                        # time slices--> make sure Nt divides 10 * T
        'N_test': 30,                    # number of test data to be generated
         #======PDE Parameters=======
        'save_pde_simulation': False,
        'K': 10 * 1e-9,                  # intrinsic permeability--> [mm^2]
        'mu_w': 9e-4,                    # viscosity of water --> [Pa * s]
        'alpha': 1e-2,                   # sink term --> [mm^2 / (N * s)]
        'D': 4e-6,                       # diffusion coefficient--> [mm^2 / s]
        'beta1': 5 / 60,                 # infusion rate --> [mm2 / s]
        'beta2': .25 * (5 / 60),         # transport gauss coeff, c_fuid * beta1 --> [mmol/s]
        'SUPG': False,                   # SUPG for transport
    }
    model_config = {'branch_layers': [problem_config['m']**2, 128, 128, 128, 128],
                    'trunk_layers': [problem_config['DIM'], 128, 128, 128, 128],
                    'num_epoch': 300000,       # number of epochs
                     # lr(n) = initial_lr * (decay_rate)^(n/decay_steps)
                    'initial_lr': 1e-3,
                    'decay_steps': 10000,
                    'decay_rate': 0.97,
                    'batch_size': 200
                    }



    # initiate PI-DeepOnet model
    model = PI_DeepONet(model_config, problem_config)

    # monitor test loss for a single instance during training
    u_test, y_test, s_test, gauss_params = generate_one_test_data(testing_rng, problem_config)
    #u_test, y_test, s_test, gauss_params = load_one_test_data(problem_config)

    if problem_config['TRAIN_MODE']:

        # ===============GENERATE TRAINING DATA===================
        generate_training_data_with_chunks(master_rng, problem_config)

        # ================LOAD TRAINING DATA======================
        (u_ic, y_ics, s_ics,
         u_bc, y_bcs,
         u_res, y_res, s_res) = load_training_data_from_chunks(master_rng, problem_config)


        # ================BUILD DATALOADERS=======================
        ic_dataset = DataGenerator(u_ic, y_ics, s_ics, model_config['batch_size'])
        bc_dataset = DataGenerator(u_bc, y_bcs, np.zeros_like(y_bcs[:, :1]), model_config['batch_size'])
        res_dataset = DataGenerator(u_res, y_res, s_res, model_config['batch_size'])

        print(f"Memory Load After Data Loading")
        get_memory_usage()
        # ====================TRAIN AND EVALUATE==================
        #eval_full_test_steps = list(range(100000, model_config['num_epoch'] + 1, 50000))
        model.train(ic_dataset, bc_dataset, res_dataset,
                    test_data=(u_test, y_test, s_test),
                    eval_full_test_steps=[],
                    )

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

        # mean relative error over all test data with N_test data points
        mean_relative_l2, per_time_rel_l2_by_time = test_model(testing_rng, problem_config, model_config)
        print(f"mean relative error over {problem_config['N_test']} testing: {100 * mean_relative_l2:0.2f} %")
        print(f"time wise relative errors: {per_time_rel_l2_by_time}")


    else:
        print(f"=========Testing the trained model!===========")
        # get predictions
        params = load_model(model, problem_config)
        s_pred = model.predict_s(params, u_test, y_test)

        # save the Nt time steps to vtk to view in Paraview
        save_vtk_comparison(y_test, s_test, s_pred, problem_config)

        # save mean + per-gaussian y_center profiles
        plot_centerline_profiles(y_test, s_test, s_pred, gauss_params, problem_config)

        # mean relative error over all test data with N_test data points
        mean_relative_l2, per_time_rel_l2_by_time = test_model(testing_rng, problem_config, model_config)
        print(f"mean relative error over {problem_config['N_test']} testing: {100 * mean_relative_l2:0.2f} %")
        print(f"time wise relative errors: {per_time_rel_l2_by_time}")


if __name__ == '__main__':
    main()