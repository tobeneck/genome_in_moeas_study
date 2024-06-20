'''
I wand to test if the algorithm is able to find the pareto-front if I only seed corner solutions
'''

import os

import pandas as pd
import numpy as np

from tea_pymoo.tracing.t_crossover import T_Crossover, TracingTypes
from pymoo.operators.crossover.ux import UniformCrossover

from benchmark_helper import run_test_combinations

from test_setup import problems, n_var


#check if the output path exists
out_path = "../data/e_and_c_benchmark"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set up the sessecary parameters
seed_ind_dfs ={
    "MACO_b" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_b.csv"),
    "MACO_p=-10" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_p=-10.csv"),
    "MACO_w=shallow" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_w=shallow.csv"),
    "MACO_w=steep" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_w=steep.csv"),
    "UF1" : pd.read_csv("../data/seed_individuals/seed_individuals_UF1.csv"),
    "UF2" : pd.read_csv("../data/seed_individuals/seed_individuals_UF2.csv"),
    "UF3" : pd.read_csv("../data/seed_individuals/seed_individuals_UF3.csv"),
    "ZDT1" : pd.read_csv("../data/seed_individuals/seed_individuals_ZDT1.csv"),
    "ZDT2" : pd.read_csv("../data/seed_individuals/seed_individuals_ZDT2.csv"),
    "ZDT3" : pd.read_csv("../data/seed_individuals/seed_individuals_ZDT3.csv"),
    "DTLZ1" : pd.read_csv("../data/seed_individuals/seed_individuals_DTLZ1.csv"),
    "DTLZ2" : pd.read_csv("../data/seed_individuals/seed_individuals_DTLZ2.csv"),
    "DTLZ3" : pd.read_csv("../data/seed_individuals/seed_individuals_DTLZ3.csv"),
}

random_populations:dict = {
    "MACO_b" : np.loadtxt("../data/initial_populations/random_pop_MACO_b_ds10_do2.csv", delimiter=",", dtype=float),
    "MACO_p=-10" : np.loadtxt("../data/initial_populations/random_pop_MACO_p=-10_ds10_do2.csv", delimiter=",", dtype=float),
    "MACO_w=shallow" : np.loadtxt("../data/initial_populations/random_pop_MACO_w=shallow_ds10_do2.csv", delimiter=",", dtype=float),
    "MACO_w=steep" : np.loadtxt("../data/initial_populations/random_pop_MACO_w=steep_ds10_do2.csv", delimiter=",", dtype=float),
    "UF1" : np.loadtxt("../data/initial_populations/random_pop_UF1_ds10_do2.csv", delimiter=",", dtype=float),
    "UF2" : np.loadtxt("../data/initial_populations/random_pop_UF2_ds10_do2.csv", delimiter=",", dtype=float),
    "UF3" : np.loadtxt("../data/initial_populations/random_pop_UF3_ds10_do2.csv", delimiter=",", dtype=float),
    "ZDT1" : np.loadtxt("../data/initial_populations/random_pop_ZDT1_ds10_do2.csv", delimiter=",", dtype=float),
    "ZDT2" : np.loadtxt("../data/initial_populations/random_pop_ZDT2_ds10_do2.csv", delimiter=",", dtype=float),
    "ZDT3" : np.loadtxt("../data/initial_populations/random_pop_ZDT3_ds10_do2.csv", delimiter=",", dtype=float),
    "DTLZ1" : np.loadtxt("../data/initial_populations/random_pop_DTLZ1_ds10_do3.csv", delimiter=",", dtype=float),
    "DTLZ2" : np.loadtxt("../data/initial_populations/random_pop_DTLZ2_ds10_do3.csv", delimiter=",", dtype=float),
    "DTLZ3" : np.loadtxt("../data/initial_populations/random_pop_DTLZ3_ds10_do3.csv", delimiter=",", dtype=float),
}

combinations_d2=[
        ["r1", "r2", "r3"],
        ["e1", "r2", "r3"],
        ["e2", "r2", "r3"],
        ["c", "r2", "r3"],
        ["e1", "e2", "r3"],
        ["e1", "e2", "c"],
    ]
combinations_d3=[
        ["r1", "r2", "r3", "r4"],
        ["e1", "r2", "r3", "r4"],
        ["e2", "r2", "r3", "r4"],
        ["e3", "r2", "r3", "r4"],
        ["c", "r2", "r3", "r4"],
        ["e1", "e2", "e3", "r4"],
        ["e1", "e2", "e3", "c"],
    ]

combinations = {
    "MACO_b" : combinations_d2,
    "MACO_p=-10" : combinations_d2,
    "MACO_w=shallow" : combinations_d2,
    "MACO_w=steep" : combinations_d2,
    "UF1" : combinations_d2,
    "UF2" : combinations_d2,
    "UF3" : combinations_d2,
    "ZDT1" : combinations_d2,
    "ZDT2" : combinations_d2,
    "ZDT3" : combinations_d2,
    "DTLZ1" : combinations_d3,
    "DTLZ2" : combinations_d3,
    "DTLZ3" : combinations_d3,
}

#actually run the tests:
run_test_combinations(
    problems=problems,
    seed_ind_dfs=seed_ind_dfs,
    output_folder=out_path,
    random_populations=random_populations,
    algorithms=["NSGA2", "MOEAD"],
    crossovers= {
    "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR),
    },
    combinations=combinations
)