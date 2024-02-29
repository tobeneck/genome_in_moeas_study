import os
import numpy as np
import pandas as pd
from pathlib import Path

#algorithmic configuration;
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.pm import PM

#benchmarking problems:
from pymoo.problems.many.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from pymoo.problems.multi.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, ZDT5
from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from problems.MACO import MultiAgentCoordinationProblem

#check if the output path exists
out_path = "../data/benchmark_variable_range"
if not os.path.exists(out_path):
    os.makedirs(out_path)

n_var = 10

problems = {
    #dtlz
    "dtlz1": DTLZ1(n_var=n_var, n_obj=3),
    "dtlz2": DTLZ2(n_var=n_var, n_obj=3),
    "dtlz3": DTLZ3(n_var=n_var, n_obj=3),
    "dtlz4": DTLZ4(n_var=n_var, n_obj=3),
    "dtlz5": DTLZ5(n_var=n_var, n_obj=3),
    "dtlz6": DTLZ6(n_var=n_var, n_obj=3),
    "dtlz7": DTLZ7(n_var=n_var, n_obj=3),

    #zdt
    "zdt1": ZDT1(n_var=n_var),
    "zdt2": ZDT2(n_var=n_var),
    "zdt3": ZDT3(n_var=n_var),
    "zdt4": ZDT4(n_var=n_var),
    #"zdt5": ZDT5(),
    "zdt6": ZDT6(n_var=n_var),

    #wfg
    "wfg1": WFG1(n_var=n_var, n_obj=3, k=4),
    "wfg2": WFG2(n_var=n_var, n_obj=3, k=4),
    "wfg3": WFG3(n_var=n_var, n_obj=3, k=4),
    "wfg4": WFG4(n_var=n_var, n_obj=3, k=4),
    "wfg5": WFG5(n_var=n_var, n_obj=3, k=4),
    "wfg6": WFG6(n_var=n_var, n_obj=3, k=4),
    "wfg7": WFG7(n_var=n_var, n_obj=3, k=4),
    "wfg8": WFG8(n_var=n_var, n_obj=3, k=4),
    "wfg9": WFG9(n_var=n_var, n_obj=3, k=4),

    #maco
    "maco_basic": MultiAgentCoordinationProblem(n_var=n_var, p=-np.inf, weights=None, classes=None, ctype=None, wtype=None),
    "maco_p-normal": MultiAgentCoordinationProblem(n_var=n_var, p=2, weights=None, classes=None, ctype=None, wtype=None),
    "maco_weights": MultiAgentCoordinationProblem(n_var=n_var, p=-np.inf, weights=None, classes=None, ctype=None, wtype="steep"),
    "maco_classes": MultiAgentCoordinationProblem(n_var=n_var, p=-np.inf, weights=None, classes=None, ctype="half", wtype=None),
}


pop_size = 91 #same as with ref dirs
sampling = FloatRandomSampling()
crossover = UX(prob=0.9)
mutation = PM(eta=20)

def get_algorithm(algorithm_name, n_obj):
    if n_obj == 2:
        ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=90) # this is to insure pop_size of 91 in all runs.
    elif n_obj == 3:
        ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=12)
    if algorithm_name == "nsga2":
        return NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
    elif algorithm_name == "moead":
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
    elif algorithm_name == "nsga3":
        return NSGA3(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
    else:
        raise ValueError("algorithm_name not recognized")

algorithm_names = ["nsga2", "moead", "nsga3"]

for problem_name, problem in problems.items():
    for algorithm_name in algorithm_names:
        print("processing", problem_name, "with", algorithm_name)

        algorithm= get_algorithm(algorithm_name, problem.n_obj)
        res = minimize(problem,
                algorithm,
                termination=('n_eval', 40000),
                seed=1,
                save_history=False,
                verbose=False)
        
        #save the genome:
        df = pd.DataFrame(res.X)
        df.insert(0, "problem", problem_name)
        df.insert(0, "algorithm", algorithm_name)
        output_filename = Path(out_path) / str("moea_benchmark_genome.csv")
        df.to_csv(output_filename, mode='a', index=False, header=(not os.path.exists(output_filename)) )

        # #save the fitness:
        # df = pd.DataFrame(res.F)
        # df.insert(0, "problem", problem_name)
        # df.insert(0, "algorithm", algorithm_name)
        # output_filename = Path(out_path) / str("moea_benchmark_objectives.csv")
        # df.to_csv(output_filename, mode='a', index=False, header=(not os.path.exists(output_filename)) )