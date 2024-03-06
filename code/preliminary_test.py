'''
I wand to test if the algorithm is able to find the pareto-front if I only seed corner solutions
'''

import os

import pandas as pd
import numpy as np

#import MACO, ZDT and DTLZ:
from pymoo.problems.multi.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pymoo.problems.many.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from problems.MACO import MultiAgentCoordinationProblem

from tea_pymoo.tracing.t_crossover import T_Crossover, TracingTypes
from pymoo.operators.crossover.ux import UniformCrossover

from benchmark_helper import run_tests

#check if the output path exists
out_path = "../data/preliminary_test"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set up the sessecary parameters
xu = 1.0
xl = 0.0

n_var = 10
n_obj_dtlz = 3

problems = {
    "MACO" : MultiAgentCoordinationProblem(n_var = n_var),
    "ZDT1" : ZDT1(n_var = n_var),
    "ZDT2" : ZDT2(n_var = n_var),
    "ZDT3" : ZDT3(n_var = n_var),
    "DTLZ1" : DTLZ1(n_var = n_var, n_obj = n_obj_dtlz),
    "DTLZ2" : DTLZ2(n_var = n_var, n_obj = n_obj_dtlz),
    "DTLZ3" : DTLZ3(n_var = n_var, n_obj = n_obj_dtlz),
}

seed_ind_dfs ={
    "MACO" : pd.DataFrame(np.zeros(n_var)).T,
    "ZDT1" : pd.DataFrame(np.zeros(n_var)).T,
    "ZDT2" : pd.DataFrame(np.zeros(n_var)).T,
    "ZDT3" : pd.DataFrame(np.zeros(n_var)).T,
    "DTLZ1" : pd.DataFrame(np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])).T,
    "DTLZ2" : pd.DataFrame(np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])).T,
    "DTLZ3" : pd.DataFrame(np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])).T,
} 

#actually run the tests:
run_tests(
    problems=problems,
    seed_ind_dfs=seed_ind_dfs,
    output_folder=out_path,
    algorithms=["NSGA2", "MOEAD"],
    crossovers= {
    "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR),
    },
)