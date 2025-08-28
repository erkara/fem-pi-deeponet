import os
def get_paths(problem_config):
    """
    This governs all project structure
    """
    model_name = problem_config['model_name']
    base_dir = f'artifacts/{model_name}'

    paths = {
        'train_data_dir':     f'{base_dir}/training_data/',
        'test_data_dir':      f'{base_dir}/testing_data/',
        'model_dir':          f'{base_dir}/models/',
        'train_results_dir':  f'{base_dir}/train_results/',
        'vtk_dir':    f'{base_dir}/vtk_results/',
        'mesh_dir':           f'{base_dir}/mesh_files/',
        'pde_simulation_dir': f'{base_dir}/pde_simulation_results/',
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths
