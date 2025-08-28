import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ['WANDB_MODE'] = 'disabled'
import wandb
import sys
import time
import json
import numpy as npo
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, clear_caches
from jax.example_libraries import optimizers
from jax.nn import relu, elu, tanh
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange, tqdm
from .config import get_paths
from .test_data_generation import generate_test_data,load_one_test_data



class DataGenerator:
    """
    Fully GPU-resident JAX data generator.
    - Preloads (u, y, s) to device once.
    - Uses jax.random.choice to sample batches directly on device.
    - No per-batch device_put or NumPy involved after init.
    TODO: improve this
    """

    def __init__(self, u_host, y_host, s_host, batch_size=64, rng_key=random.PRNGKey(0)):
        """
        Parameters:
            u_host, y_host, s_host: NumPy arrays on RAM
        """
        # move all to GPU
        self.u = jax.device_put(u_host)
        self.y = jax.device_put(y_host)
        self.s = jax.device_put(s_host)

        self.N = self.u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        """
        Sample a batch on GPU and returns it directly.
        """
        self.key, subkey = random.split(self.key)
        idx = random.choice(subkey, self.N, shape=(self.batch_size,), replace=True)

        u_batch = self.u[idx]
        y_batch = self.y[idx]
        s_batch = self.s[idx]

        return (u_batch, y_batch), s_batch



def modified_deeponet(branch_layers, trunk_layers, activation=tanh):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    def init(rng_key1, rng_key2):
        U1, b1 = xavier_init(random.PRNGKey(12345), branch_layers[0], branch_layers[1])
        U2, b2 = xavier_init(random.PRNGKey(54321), trunk_layers[0], trunk_layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        key2, *keys2 = random.split(rng_key2, len(trunk_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        trunk_params = list(map(init_layer, keys2, trunk_layers[:-1], trunk_layers[1:]))
        return (branch_params, trunk_params, U1, b1, U2, b2)

    def apply(params, u, y):
        branch_params, trunk_params, U1, b1, U2, b2 = params
        U = activation(np.dot(u, U1) + b1)
        V = activation(np.dot(y, U2) + b2)
        for k in range(len(branch_layers) - 2):
            W_b, b_b = branch_params[k]
            W_t, b_t = trunk_params[k]

            B = activation(np.dot(u, W_b) + b_b)
            T = activation(np.dot(y, W_t) + b_t)

            u = np.multiply(B, U) + np.multiply(1 - B, V)
            y = np.multiply(T, U) + np.multiply(1 - T, V)

        W_b, b_b = branch_params[-1]
        W_t, b_t = trunk_params[-1]
        B = np.dot(u, W_b) + b_b
        T = np.dot(y, W_t) + b_t
        outputs = np.sum(B * T)
        return outputs

    return init, apply


class PI_DeepONet:
    def __init__(self, model_config,problem_config):
        self.branch_layers = model_config['branch_layers']
        self.trunk_layers = model_config['trunk_layers']
        self.init, self.apply = modified_deeponet(self.branch_layers, self.trunk_layers, activation=np.tanh)
        params = self.init(rng_key1=random.PRNGKey(1234), rng_key2=random.PRNGKey(4321))

        # implement exponential decay scheme
        # TODO: replace with ReduceLROnPlateau style idea?
        lr = optimizers.exponential_decay(model_config['initial_lr'], decay_steps=model_config['decay_steps'],
                                          decay_rate=model_config['decay_rate'])
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params)
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # logs to track
        self.logs = {
            "step": [],
            "loss_total": [],
            "loss_ic": [],
            "loss_neumann": [],
            "loss_res": [],
            "test_l2": []
        }

        # parameter configuration
        self.D = problem_config['D']
        self.beta2 = problem_config['beta2']
        self.N_test = problem_config['N_test']
        self.testing_rng = npo.random.default_rng(seed=3451)
        self.test_data_cache = None
        self.num_epochs = model_config['num_epoch']
        self.log_int = problem_config['log_int']

        # maintain a clean structure
        self.model_name = problem_config['model_name']
        self.artifact_dir = f"artifacts/{self.model_name}"
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.problem_config = problem_config
        self.model_config = model_config

    # Define the opeartor net
    def operator_net(self, params, u, x, y, t):
        # x, y, and t as input coordinates
        input_coords = np.stack([x, y, t])
        # attach Branch input
        outputs = self.apply(params, u, input_coords)
        return outputs

    # Define the residual net
    def residual_net(self, params, u, x, y, t, ux, v1, v2, dv1dx, dv2dy):
        # Diffusion coefficient--> D(x,y) in the future
        D = self.D
        beta2 = self.beta2

        # Compute the predicted solution s(x, y, t)
        s = self.operator_net(params, u, x, y, t)  # Predicted solution s(x, y, t)

        # Compute time and space derivatives of s
        s_t = grad(self.operator_net, argnums=4)(params, u, x, y, t)  # ds/dt
        s_x = grad(self.operator_net, argnums=2)(params, u, x, y, t)  # ds/dx
        s_y = grad(self.operator_net, argnums=3)(params, u, x, y, t)  # ds/dy
        s_xx = grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, y, t)  # d2s/dx2
        s_yy = grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, x, y, t)  # d2s/dy2

        # Compute the residual for the convection-diffusion, note we move things to the left
        res = s_t - D * (s_xx + s_yy) + ((dv1dx + dv2dy) * s + v1 * s_x  + v2 * s_y) - beta2 * ux

        return res

    ###=======================================NEW STUFF====================
    @partial(jit, static_argnums=(0,))
    def loss_ic(self, params, batch):
        inputs, outputs = batch
        u, coords = inputs  # coords: (P_ics, 3)
        x, y, t = coords[:, 0], coords[:, 1], coords[:, 2]

        s_pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, u, x, y, t)
        loss = np.mean((s_pred.flatten() - outputs.flatten()) ** 2)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss_neumann(self, params, batch):
        inputs, _ = batch
        u, coords = inputs
        x, y, t = coords[:, 0], coords[:, 1], coords[:, 2]

        # Determine normal vector components based on rectangular domain
        _, xmax = self.problem_config['space_domain'][0]
        _, ymax = self.problem_config['space_domain'][1]

        tol = 1e-4
        nx = np.where(np.abs(x - 0.0) < tol, -1.0,
                      np.where(np.abs(x - xmax) < tol, 1.0, 0.0))
        ny = np.where(np.abs(y - 0.0) < tol, -1.0,
                      np.where(np.abs(y - ymax) < tol, 1.0, 0.0))

        #from jax import debug
        #debug.print("\nx nonzero count: {} / {}", np.count_nonzero(nx), nx.shape[0])
        #debug.print("\ny nonzero count: {} / {}", np.count_nonzero(ny), ny.shape[0])

        # Compute directional derivative (n * grad(c))
        def flux_fn(params, u, x, y, t, nx, ny):
            s_x = grad(self.operator_net, argnums=2)(params, u, x, y, t)
            s_y = grad(self.operator_net, argnums=3)(params, u, x, y, t)
            return nx * s_x + ny * s_y

        flux = vmap(flux_fn, (None, 0, 0, 0, 0, 0, 0))(params, u, x, y, t, nx, ny)
        loss = np.mean(flux ** 2)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        # Fetch dataset from batch
        inputs, outputs = batch
        u, input_coords = inputs
        # Extract space-time points and other variables
        x = input_coords[:, 0]  # x_r
        y = input_coords[:, 1]  # y_r
        t = input_coords[:, 2]  # t_r
        ux = input_coords[:, 3]  # f(x, y, t)
        v1 = input_coords[:, 4]  # velocity component 1
        v2 = input_coords[:, 5]  # velocity component 2
        dv1dx = input_coords[:, 6]  # dv1/dx
        dv2dy = input_coords[:, 7]  # dv2/dy

        # Vectorized residual prediction
        pred = vmap(self.residual_net, (None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
            params, u, x, y, t, ux, v1, v2, dv1dx, dv2dy
        )

        # Residual loss = mean squared PDE residual
        return np.mean(pred ** 2)

    def loss(self, params, ic_batch, bc_batch, res_batch):
        loss_ic = self.loss_ic(params, ic_batch)
        loss_bc = self.loss_neumann(params, bc_batch)
        loss_res = self.loss_res(params, res_batch)
        lam_ic, lam_bc, lam_res = 1.0, 1e-3, 10.0
        return lam_ic * loss_ic + lam_bc * loss_bc + lam_res * loss_res

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ic_batch, bc_batch, res_batch):
        params = self.get_params(opt_state)
        grads = grad(self.loss)(params, ic_batch, bc_batch, res_batch)
        return self.opt_update(i, grads, opt_state)


    def train(self, ic_dataset, bc_dataset, res_dataset, test_data, eval_full_test_steps=None):
        """
        main training loop
        """
        ic_data = iter(ic_dataset)
        bc_data = iter(bc_dataset)
        res_data = iter(res_dataset)

        # Unpack test data
        u_test, y_test, s_test = test_data

        # Main training loop
        start_time = time.time()
        pbar = trange(self.num_epochs + 1)

        for it in pbar:
            ic_batch = next(ic_data)
            bc_batch = next(bc_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, ic_batch, bc_batch, res_batch)

            # Logging and diagnostics
            if it % self.log_int == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ic_batch, bc_batch, res_batch)
                loss_ic_val = self.loss_ic(params, ic_batch)
                loss_bc_val = self.loss_neumann(params, bc_batch)
                loss_res_val = self.loss_res(params, res_batch)

                # Test prediction
                s_pred = self.predict_s(params, u_test, y_test).reshape(s_test.shape)
                rel_l2_error = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)

                # Logging
                self.logs["step"].append(it)
                self.logs["loss_total"].append(float(loss_value))
                self.logs["loss_ic"].append(float(loss_ic_val))
                self.logs["loss_neumann"].append(float(loss_bc_val))
                self.logs["loss_res"].append(float(loss_res_val))
                self.logs["test_l2"].append(float(rel_l2_error))


                #status bar
                pbar.set_postfix({
                    'loss': f"{loss_value:.4e}",
                    'ic': f"{loss_ic_val:.4e}",
                    'neumann': f"{loss_bc_val:.4e}",
                    'res': f"{loss_res_val:.4e}",
                    'test_l2(%)': f"{100 * rel_l2_error:.1f}"
                })

                # Crash early if nan is detected
                for name, val in {
                    "loss": loss_value,
                    "loss_ic": loss_ic_val,
                    "loss_neumann": loss_bc_val,
                    "loss_res": loss_res_val
                }.items():
                    if not np.isfinite(val):
                        print(f"NaN detected in {name} at step {it}.")
                        sys.exit(1)

            # Optional full test eval over N_test samples
            if eval_full_test_steps is not None and it in eval_full_test_steps:
                if self.test_data_cache is None:
                    generate_test_data(self.testing_rng, self.problem_config)
                    self.test_data_cache = [
                        load_one_test_data(self.problem_config, i)
                        for i in range(self.N_test)
                    ]

                params = self.get_params(self.opt_state)
                rel_l2s = []
                for u_test, y_test, s_test, _ in self.test_data_cache:
                    s_pred = self.predict_s(params, u_test, y_test).reshape(s_test.shape)
                    rel_l2 = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
                    rel_l2s.append(rel_l2)

                mean_l2 = float(100 * npo.mean(npo.array(rel_l2s)))
                self.logs.setdefault("mean_l2_vals", []).append({"step": it, "mean_l2": round(mean_l2, 2)})
                print(f"\n[MEAN L2 over {self.N_test} samples @ step {it}] = {mean_l2:.2f}%")
        print(f"==================================================")

        #===save the model=====
        paths = get_paths(self.problem_config)
        flat_params, _ = ravel_pytree(self.get_params(self.opt_state))
        model_path = os.path.join(paths['model_dir'], "pi_deeponet.npy")
        np.save(model_path, flat_params)
        print(f"Trainin over, model saved at : {model_path}")

        #======save training logs===
        elapsed_time_min = (time.time() - start_time) / 60
        self.logs["training_time_min"] = round(elapsed_time_min,2)
        log_file = f"{paths['train_results_dir']}/train_log.json"
        with open(log_file, "w") as f:
            json.dump(self.logs, f, indent=2)

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1], Y_star[:, 2])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0, 0, 0, 0, 0, 0))(
            params, U_star, Y_star[:, 0], Y_star[:, 1], Y_star[:, 2],  # x, y, t
            Y_star[:, 3],  # ux
            Y_star[:, 4], Y_star[:, 5],  # v1,v2
            Y_star[:, 6], Y_star[:, 7]  # dv1/dx, dv2/dy
        )

        return r_pred









