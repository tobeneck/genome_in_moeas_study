'''
This file contains preliminary test data for the tests with the Sphere function.
'''
import os

import pandas as pd
import numpy as np

from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.tracing.tracing_types import TracingTypes

from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover

from problems.sphere import Sphere
from operators.linear_crossover_gene import LXg
from operators.linear_crossover_ind import LXi

from benchmark_helper import run_tests

#check if the output path exists
out_path = "../data/sphere_function_data"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set the randon seed
np.random.seed(1)


#set up the sessecary parameters
xu = 50
xl = -50
tracing_type = TracingTypes.TRACE_VECTOR

problems = {
    4 : Sphere(n_var = 4, xu=xu, xl=xl),
    8 : Sphere(n_var = 8, xu=xu, xl=xl),
    16 : Sphere(n_var = 16, xu=xu, xl=xl),
    32 : Sphere(n_var = 32, xu=xu, xl=xl),
}

crossovers = {
        "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=tracing_type),
        "SBX" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=tracing_type),
        "LXg" : T_Crossover(crossover=LXg(), tracing_type=tracing_type),
        "LXi" : T_Crossover(crossover=LXi(), tracing_type=tracing_type), #this should bias no optimal genetic material the most!
    }

seed_ind_dfs = {
    4 : pd.read_csv("../data/seed_individuals/seed_individuals_Sphere_4.csv"),
    8 : pd.read_csv("../data/seed_individuals/seed_individuals_Sphere_8.csv"),
    16 : pd.read_csv("../data/seed_individuals/seed_individuals_Sphere_16.csv"),
    32 : pd.read_csv("../data/seed_individuals/seed_individuals_Sphere_32.csv"),
}


#actually run the tests:
run_tests(
    problems=problems,
    crossovers=crossovers,
    seed_ind_dfs=seed_ind_dfs,
    output_folder=out_path,
)