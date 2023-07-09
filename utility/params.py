import yaml

def load_params_yml(params_path):
    with open(params_path) as params_file:
        params = yaml.safe_load(params_file)
    return params