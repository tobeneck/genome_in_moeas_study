'''
I wand to test if the algorithm is able to find the pareto-front if I only seed corner solutions
'''

import os

import pandas as pd
import numpy as np

#import MACO, ZDT and DTLZ:
from pymoo.problems.multi.zdt import ZDT1, ZDT2, ZDT3
from pymoo.problems.many.dtlz import DTLZ1, DTLZ2, DTLZ3
from pymoo_problems.moo.maco import MACO

from tea_pymoo.tracing.t_crossover import T_Crossover, TracingTypes
from pymoo.operators.crossover.ux import UniformCrossover

from benchmark_helper import run_test_combinations

#check if the output path exists
out_path = "../data/e_and_c_benchmark"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set up the sessecary parameters
xu = 1.0
xl = 0.0

n_var = 10
n_obj_dtlz = 3

problems = {
    "MACO_b" : MACO(n_var = n_var),
    "MACO_p=-10" : MACO(n_var = n_var, p=-10),
    "MACO_w=shallow" : MACO(n_var = n_var, wtype="shallow"),
    "MACO_w=steep" : MACO(n_var = n_var, wtype="steep"),
    "ZDT1" : ZDT1(n_var = n_var),
    "ZDT2" : ZDT2(n_var = n_var),
    "ZDT3" : ZDT3(n_var = n_var),
    "DTLZ1" : DTLZ1(n_var = n_var, n_obj = n_obj_dtlz),
    "DTLZ2" : DTLZ2(n_var = n_var, n_obj = n_obj_dtlz),
    "DTLZ3" : DTLZ3(n_var = n_var, n_obj = n_obj_dtlz),
}

seed_ind_dfs ={
    "MACO_b" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_b.csv"),
    "MACO_p=-10" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_p=-10.csv"),
    "MACO_w=shallow" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_w=shallow.csv"),
    "MACO_w=steep" : pd.read_csv("../data/seed_individuals/seed_individuals_MACO_w=steep.csv"),
    "ZDT1" : pd.read_csv("../data/seed_individuals/seed_individuals_ZDT1.csv"),
    "ZDT2" : pd.read_csv("../data/seed_individuals/seed_individuals_ZDT2.csv"),
    "ZDT3" : pd.read_csv("../data/seed_individuals/seed_individuals_ZDT3.csv"),
    "DTLZ1" : pd.read_csv("../data/seed_individuals/seed_individuals_DTLZ1.csv"),
    "DTLZ2" : pd.read_csv("../data/seed_individuals/seed_individuals_DTLZ2.csv"),
    "DTLZ3" : pd.read_csv("../data/seed_individuals/seed_individuals_DTLZ3.csv"),
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
    algorithms=["NSGA2", "MOEAD"],
    crossovers= {
    "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR),
    },
    combinations=combinations
)