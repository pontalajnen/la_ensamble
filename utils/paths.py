import os


ROOT = os.getcwd()

# local storage of compute node on cluster
LOCAL_STORAGE = "/local_storage/users/kadec/"

# name of folder where data is (can be combined with either ROOT or LOCAL_STORAGE)
DATA_DIR = "/data/"

# depends on where models are stored; had to move this due to storage issues
MODEL_PATH = "/experiment_results/"
MODEL_PATH_LOCAL = "/local_storage/users/kadec/experiment_results/"

# where the results of evaluate.py are stored
RESULT_DIR = "/experiment_results/table_metrics/"
