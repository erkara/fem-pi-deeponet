"""
u = -K/mu * grad(p)           ---> Darcy
div(u) = -alpha * p + f(x)    ---> alpha = L_p*S/V, Sterling
dc/dt = div(D * grad(c)) - div(u * c)  + g(x) ----> Convective transport t=[0,t1]

Note: stick to native numpy in this file as Firedrake or Scipy are not well-integrated with Jax
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*enabling safe_sync as default.*")
os.environ["OMP_NUM_THREADS"] = "1"
#=======#
import gmsh
from firedrake import *
from firedrake.output import VTKFile
import time
from firedrake.__future__ import interpolate
from mpi4py import MPI
from scipy.interpolate import griddata
import numpy as npo
from tqdm import trange
import shutil
from .config import get_paths


def gaussian_source(mesh, gauss_params):
    """
    Generates a normalized sum of multiple Gaussians over a Firedrake mesh.

    gauss_params: dictionary with keys:
        - 'centers': list of (x0, y0)
        - 'sigma': list of corresponding sigmas
    """
    centers = gauss_params['centers']
    sigmas = gauss_params['sigma']

    V_temp = FunctionSpace(mesh, 'CG', 2)
    x = SpatialCoordinate(mesh)

    total_gaussian_expr = 0
    for (x0, y0), sigma in zip(centers, sigmas):
        x0 = npo.asarray(x0)
        y0 = npo.asarray(y0)
        sigma = npo.asarray(sigma)

        r = (x[0] - x0)**2 + (x[1] - y0)**2
        total_gaussian_expr += exp(-r / (2 * sigma**2))

    # Interpolate unnormalized sum onto mesh
    unscaled = Function(V_temp).interpolate(total_gaussian_expr)

    # Normalize to ensure int(f(x, y) dxdy) = 1
    integral = assemble(unscaled * dx)
    normalized_gaussian = Function(V_temp).interpolate((1.0 / integral) * total_gaussian_expr)

    return normalized_gaussian


def generate_refined_mesh(problem_config, gauss_params):
    """
    Generate an adaptively refined mesh for multiple Gaussians using gmsh.
    Each Gaussian gets local refinement near its center, and the total mesh is blended.
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Gaussian Mesh")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # works pretty well

    # Extract problem domain
    Lx, Ly = problem_config['space_domain'][0][1], problem_config['space_domain'][1][1]

    centers = gauss_params['centers']
    sigmas = gauss_params['sigma']

    # Mesh size parameters (per center)
    hmax = 0.04 * max(Lx, Ly)
    hmin_list = [sigma / 6 for sigma in sigmas]
    k_list = [10 + 10 * (0.001 - sigma) / 0.001 for sigma in sigmas]
    r_refine_list = [max(k * sigma, 0.1 * min(Lx, Ly)) for k, sigma in zip(k_list, sigmas)]
    transition_width_list = [r / 2 for r in r_refine_list]

    # Define rectangular domain in GMSH
    p1 = gmsh.model.geo.addPoint(0, 0, 0, hmax)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, hmax)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, hmax)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0, hmax)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # Combine decay expressions for all centers
    expr_parts = []
    for (x0, y0), r_refine, trans_w in zip(centers, r_refine_list, transition_width_list):
        expr_parts.append(
            f"0.5 * (1 + tanh((sqrt((x - {x0})^2 + (y - {y0})^2) - {r_refine}) / {trans_w}))"
        )

    # Use the smallest hmin for safe refinement
    global_hmin = min(hmin_list)
    expr = f"{global_hmin} + ({hmax} - {global_hmin}) * {' * '.join(expr_parts)}"

    field_id = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field_id, "F", expr)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

    # some mesh options
    gmsh.option.setNumber("Mesh.MeshSizeMin", global_hmin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", global_hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    gmsh.model.mesh.generate(2)


    # save mesh
    paths = get_paths(problem_config)
    mesh_dir = paths['mesh_dir']
    if os.path.exists(mesh_dir):
        shutil.rmtree(mesh_dir)
    os.makedirs(mesh_dir)
    gmsh.write(f"{mesh_dir}/refined_mesh.msh")
    gmsh.finalize()

    # Read refined mesh for Firedrake
    mesh = Mesh(f"{mesh_dir}/refined_mesh.msh")

    return mesh



def solve_darcy(problem_config, gauss_params):
    # generate refined mesh from gmesh based on the sample Gaissuan
    mesh = generate_refined_mesh(problem_config, gauss_params)

    # solver parameters which I use since the dawn of the time
    parameters = {
        #'snes_monitor': None,
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps"
    }

    # problem paramaters
    alpha = Constant(problem_config['alpha'])  # sink term--> [mm^2 / (N * s)]
    K = Constant(problem_config['K'])          # intrinsic permeability--> [mm^2]
    mu_w = Constant(problem_config['mu_w'])    # viscosity of water --> [Pa * s]
    k = K / mu_w


    # define function spaces
    # quad mesh not implemented for BDM in Firedrake
    V = FunctionSpace(mesh, "BDM", 2)  # velocity --> Brezzi-Douglas-Marini
    Q = FunctionSpace(mesh, "DG", 1)  # pressure --> Discontinuous Galerkin

    # mixed fem space
    W = V * Q
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # variational form for Darcy
    beta1 = problem_config['beta1']
    f = gaussian_source(mesh, gauss_params)
    a = (dot(u, v) - dot(p, div(k * v)) + div(u) * q + alpha * p * q) * dx
    L = beta1 * f * q * dx
    sol_up = Function(W)

    # solve Darcy with zero pressure boundary
    bcs_p = DirichletBC(W.sub(1), Constant(0.), 'on_boundary')
    solve(a == L, sol_up, bcs=[bcs_p], solver_parameters=parameters)
    u, p = sol_up.subfunctions

    ## some debugging stuff
    V_scalar = FunctionSpace(mesh, 'CG', 2)
    dv1_dx = Function(V_scalar).project(u[0].dx(0),name='dv1_dx')
    dv2_dy = Function(V_scalar).project(u[1].dx(1),name='dv2_dy')

    ## check where we are in convection-domination
    u_norm = sqrt(dot(u, u))
    Pe = 0.5 * u_norm * CellSize(mesh) / problem_config['D']
    Pe_values = Function(Q).interpolate(Pe)

    # # print max quantities
    # u_norm_func = Function(Q).interpolate(u_norm)
    # max_u_norm = max(u_norm_func.dat.data_ro)
    # cell_size_func = Function(Q).interpolate(CellSize(mesh))
    # max_cell_size = max(cell_size_func.dat.data_ro)
    # print(f"Maximum u_norm: {max_u_norm}")
    # print(f"Maximum cell size: {max_cell_size}")
    # print(f"max peclet:{max(Pe_values.dat.data_ro)}")

    ## vtk stuff--> comment once sure it works as expected
    if problem_config['save_pde_simulation']:
        paths = get_paths(problem_config)
        pde_simulation_dir = paths['pde_simulation_dir']
        vtkfile_up = VTKFile(f"{pde_simulation_dir}/solution_up.pvd")
        p.rename('pressure')
        u.rename('velocity')
        Pe_values.rename('peclet')
        vtkfile_up.write(p, u, dv1_dx, dv2_dy, Pe_values)

    return u, mesh

def mapped_velocity_fem(problem_config, gauss_params, xr, yr):
    # generate an adaptive mesh and solve Darcy to get u
    u, mesh = solve_darcy(problem_config, gauss_params)

    # extract velocity and some derivatives
    V_scalar = FunctionSpace(mesh, 'CG', 1)
    v1 = Function(V_scalar).project(u[0])
    v2 = Function(V_scalar).project(u[1])
    dv1_dx = Function(V_scalar).project(v1.dx(0))
    dv2_dy = Function(V_scalar).project(v2.dx(1))

    # Get the values at mesh vertex coordinates
    coords = mesh.coordinates.dat.data_ro
    v1_values = v1.dat.data_ro
    v2_values = v2.dat.data_ro
    dv1_dx_values = dv1_dx.dat.data_ro
    dv2_dy_values = dv2_dy.dat.data_ro

    # Use griddata to interpolate quantities onto the new grid
    v1_r = griddata(coords, v1_values.flatten(), (xr, yr), method='cubic')
    v2_r = griddata(coords, v2_values.flatten(), (xr, yr), method='cubic')
    dv1_dx_r = griddata(coords, dv1_dx_values.flatten(), (xr, yr), method='cubic')
    dv2_dy_r = griddata(coords, dv2_dy_values.flatten(), (xr, yr), method='cubic')

    return v1_r, v2_r, dv1_dx_r, dv2_dy_r


def solve_convection_diffusion(gauss_params, problem_config,show_progress=False):
    parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps"
    }


    # some params
    dt = 1
    T = problem_config['time_domain'][1]
    Nt = problem_config['Nt']
    num_steps = int(T / dt)
    step_interval = num_steps // Nt

    D = Constant(problem_config['D'])
    u, mesh = solve_darcy(problem_config, gauss_params)
    start_time = time.time()
    C = FunctionSpace(mesh, "CG", 1)
    w = TestFunction(C)
    c = Function(C)

    # Initialization for the time-stepping
    t = Constant(0.0)
    c_n = Function(C).interpolate(Constant(0.0))

    # Classical variational formulation for transport
    beta2 = problem_config['beta2']
    g = gaussian_source(mesh, gauss_params)
    F = (inner((c - c_n) / dt, w) + inner(D * grad(c), grad(w)) - inner(u * c, grad(w)) - beta2 * g * w) * dx

    if problem_config['SUPG']:
        # some supg stuff
        u_norm = sqrt(dot(u, u))
        h = CellSize(mesh)
        residual = (c - c_n) / dt - div(D * grad(c)) + div(u * c) - beta2 * g
        tau = h / (2 * u_norm)
        SUPG_term = tau * dot(u, grad(w)) * residual * dx
        F += SUPG_term

    # VTK output, disable I/O to speed up simulation
    if problem_config['save_pde_simulation']:
        paths = get_paths(problem_config)
        pde_simulation_dir = paths['pde_simulation_dir']
        vtkfile = VTKFile(f"{pde_simulation_dir}/solution_concentration.pvd")

    # store the solution at specified output times
    c_solutions = []
    current_time = dt
    step_range = trange(1, num_steps + 1) if show_progress else range(1, num_steps + 1)

    for step in step_range:
        # Update time
        t.assign(current_time)
        solve(F == 0, c, solver_parameters=parameters)

        # Only save a snapshot if the current time matches one of the output times
        if step % step_interval == 0:
            snapshot = c.copy(deepcopy=True)
            c_solutions.append(snapshot)

        #Write VTK output for visualization
        c.rename('concentration')
        if problem_config['save_pde_simulation']:
            vtkfile.write(c, time=current_time)

        # update
        c_n.assign(c)
        current_time += dt

    # make sure the number of stored solutions matches Nt
    assert len(c_solutions) == Nt, f"Expected {Nt} solutions, but got {len(c_solutions)}"
    simulation_time = time.time() - start_time
    return c_solutions, mesh, simulation_time

def solve_cd_for_speed_test(gauss_params, problem_config):
    # helper function for run-time comparisons
    # due to small problem size, it does not benefit from parallelization
    parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps"
    }


    # Fixed time step size
    dt = 1
    # Total time and number of output slices
    T = problem_config['time_domain'][1]

    # Calculate the number of time steps needed to reach T
    num_steps = int(T / dt)           # e.g. T=50/1 = 50

    D = Constant(problem_config['D'])
    u, mesh = solve_darcy(problem_config, gauss_params)
    start_time = time.time()
    C = FunctionSpace(mesh, "CG", 1)
    w = TestFunction(C)
    c = Function(C)

    # Initialization for the time-stepping
    t = Constant(0.0)
    c_n = Function(C).interpolate(Constant(0.0))

    # Classical variational formulation for transport
    beta2 = problem_config['beta2']
    g = gaussian_source(mesh, gauss_params)
    F = (inner((c - c_n) / dt, w) + inner(D * grad(c), grad(w)) - inner(u * c, grad(w)) - beta2 * g * w) * dx

    if problem_config['SUPG']:
        # some supg stuff
        u_norm = sqrt(dot(u, u))
        h = CellSize(mesh)
        residual = (c - c_n) / dt - div(D * grad(c)) + div(u * c) - g
        tau = h / (2 * u_norm)

        SUPG_term = tau * dot(u, grad(w)) * residual * dx
        F += SUPG_term

    # store the solution at specified output times
    current_time = dt
    for step in range(1, num_steps + 1):
        # update
        t.assign(current_time)
        solve(F == 0, c, solver_parameters=parameters)
        c.rename('concentration')
        c_n.assign(c)
        current_time += dt

    simulation_time = time.time() - start_time
    return simulation_time



def map_concentration_fem(gauss_params, problem_config, xr, yr):
    # Solve the convection-diffusion problem and get concentration `c` and mesh
    c_solutions, mesh, _ = solve_convection_diffusion(gauss_params, problem_config)

    # Scalar FunctionSpace on the mesh
    V_scalar = FunctionSpace(mesh, 'CG', 1)
    coords = mesh.coordinates.dat.data_ro  # (M, 2)

    # store mapped concentrations for each point at each time step
    Q = len(xr)                  # Number of points in Gauss grid from operator problem
    Nt = problem_config['Nt']    # number of times to hold the time-dependent solution
    c_sol = npo.zeros((Q, Nt))   # store solution for Q points over time steps

    c_temp = Function(V_scalar)

    # Iterate over each time step (Nt solutions in c_solutions)
    for step in range(Nt):
        # current solution
        c = c_solutions[step]
        c_temp.project(c)
        c_values = c_temp.dat.data_ro  # (M,)

        # Interpolate concentration to operator grid (xr, yr)
        c_r_flat = griddata(coords, c_values.flatten(), (xr, yr), method='cubic')

        # Store the interpolated values for this time step (1D array for each time step)
        c_sol[:, step] = c_r_flat


    return c_sol

