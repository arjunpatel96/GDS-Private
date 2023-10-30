import yaml
import logging
import os


def read_config():
    cwd = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(cwd, 'config.yml')
    content = read_yaml(config_file)
    return content


def read_yaml(yaml_file):
    content = None
    with open(yaml_file, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))
    return content


def get_data_dir():
    config = read_config()
    datadir = config['dataDirectory']
    if datadir.startswith('/'):
        return datadir
    cwd = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.abspath(os.path.join(cwd, datadir))
    return datadir
