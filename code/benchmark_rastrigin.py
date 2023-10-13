'''
This file contains preliminary test data for the tests with the Rastrigin function.
'''
import os

import pandas as pd
import numpy as np

from problems.rastrigin import Rastrigin

from benchmark_helper import run_tests

#check if the output path exists
out_path = "../data/rastrigin_function_data"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set the randon seed
np.random.seed(1)


#set up the sessecary parameters
xu = 50
xl = -50

problems = {
    4 : Rastrigin(n_var = 4, xu=xu, xl=xl),
    8 : Rastrigin(n_var = 8, xu=xu, xl=xl),
    16 : Rastrigin(n_var = 16, xu=xu, xl=xl),
    32 : Rastrigin(n_var = 32, xu=xu, xl=xl),
}

seed_ind_dfs = {
    4 : pd.read_csv("../data/seed_individuals/seed_individuals_Rastrigin_4.csv"),
    8 : pd.read_csv("../data/seed_individuals/seed_individuals_Rastrigin_8.csv"),
    16 : pd.read_csv("../data/seed_individuals/seed_individuals_Rastrigin_16.csv"),
    32 : pd.read_csv("../data/seed_individuals/seed_individuals_Rastrigin_32.csv"),
}


#actually run the tests:
run_tests(
    problems=problems,
    seed_ind_dfs=seed_ind_dfs,
    output_folder=out_path,
)