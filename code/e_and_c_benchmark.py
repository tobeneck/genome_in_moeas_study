'''
I wand to test if the algorithm is able to find the pareto-front if I only seed corner solutions
'''

import os

import numpy as np
import pandas as pd

from tea_pymoo.tracing.t_crossover import T_Crossover, TracingTypes
from pymoo.operators.crossover.ux import UniformCrossover

from benchmark_helper import run_test_combinations

from test_setup import problems, max_gen


#check if the output path exists
out_path = "../data/e_and_c_benchmark"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set up the sessecary parameters
seed_ind_dfs = dict()
for problem_name in problems.keys():
    problem=problems[problem_name]
    seed_ind_dfs[problem_name] = pd.read_csv("../data/seed_individuals/seed_individuals_"+problem_name+".csv")

random_populations = dict()
for problem_name in problems.keys():
    problem=problems[problem_name]
    random_populations[problem_name] = np.loadtxt("../data/initial_populations/random_pop_"+problem_name+"_ds"+str(problem.n_var)+"_do"+str(problem.n_obj)+".csv", delimiter=",", dtype=float)


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

combinations = dict()
for problem_name in problems.keys():
    problem=problems[problem_name]
    if problem.n_obj == 2:
        combinations[problem_name] = combinations_d2
    elif problem.n_obj == 3:
        combinations[problem_name] = combinations_d3

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
    combinations=combinations,
    n_gen=max_gen,
)