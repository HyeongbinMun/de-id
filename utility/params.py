import yaml

def load_params_yml(config):
    with open(config) as params_file:
        params = yaml.safe_load(params_file)
    return params