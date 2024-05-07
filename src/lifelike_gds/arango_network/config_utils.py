import importlib.resources as resources

import yaml


def read_config():
    with resources.open_binary('lifelike_gds.arango_network', 'config.yml') as config_file:
        content = read_yaml(config_file)
        return content


def read_yaml(yaml_file):
    try:
        return yaml.safe_load(yaml_file)
    except yaml.YAMLError as err:
        raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))
