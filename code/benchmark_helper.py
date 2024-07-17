from tea_pymoo.tracing.t_mutation import T_Mutation
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.callbacks.general.counting_impact_callback import Counting_Impact_Callback
from tea_pymoo.callbacks.general.genome_callback import Genome_Callback
from tea_pymoo.callbacks.moo.performance_indicators import Performance_Indicators_Callback
from tea_pymoo.callbacks.moo.fitness_and_ranks_genome import Fitness_and_Ranks_Callback

from tea_pymoo.callbacks.accumulated_callback import AccumulateCallbacks
from tea_pymoo.tracing.tracing_types import TracingTypes
from tea_pymoo.tracing.t_sampling import T_Sampling

from pymoo.optimize import minimize
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.util.ref_dirs import get_reference_directions

import numpy as np

def get_algorithm(algorithm_name, n_obj, pop_size, sampling, crossover, mutation):
    if n_obj == 2:
        ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=90) # this is to insure pop_size of 91 in all runs.
    elif n_obj == 3:
        ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=12)
    if algorithm_name == "NSGA2":
        return NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    elif algorithm_name == "MOEAD":
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    elif algorithm_name == "NSGA3":
        return NSGA3(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    else:
        raise ValueError("algorithm_name not recognized")

def get_reference_point(n_obj:int):
    if n_obj == 2:
        return [1.1, 1.1]
    elif n_obj == 3:
        return [1.1, 1.1, 1.1]
    else:
        raise ValueError("n_obj not recognized")
    
def run_test_combinations(
        problems:dict,
        output_folder:str,
        seed_ind_dfs:dict,
        random_populations:dict,
        crossovers:dict= {
            "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR),
            "SBX" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=TracingTypes.TRACE_VECTOR),
        },
        n_gen:int=100,
        pop_size:int=91,
        tracing_type:TracingTypes=TracingTypes.TRACE_VECTOR,
        algorithms=["NSGA2", "NSGA3", "MOEAD"],
        combinations=dict(),
        ):
    '''
    TODO!
    '''
    t_sampling = T_Sampling(FloatRandomSampling(), tracing_type=tracing_type)

    #run the tests:
    for problem_name in problems:#iterate over the problems
        problem= problems[problem_name]
        for crossover_name in crossovers:#iterate over the crossovers
            for algorithm_name in algorithms:#iterate over the algorithms
                for combination in combinations[problem_name]:#iterate over the seed individuals in this generation
                    #get the seed individuals from the list of combinations
                    seed_ind_df = seed_ind_dfs[problem_name]
                    seed_inds = np.array( [ seed_ind_df[seed_ind_df["type"] == combination[i]].iloc[0][1:problem.n_var+1].to_numpy(dtype=float) for i in range(len(combination)) ], dtype=float )

                    #build one seed tyoe string, only with non-randome types
                    seed_types = [ seed_ind_df[seed_ind_df["type"] == combination[i]].iloc[0]["type"] for i in range(len(combination)) ]
                    seed_types = [ seed_type for seed_type in seed_types if not "r" in seed_type ]
                    seed_type = "+".join(seed_types)
                    if seed_type == "":
                        seed_type = "r"

                    print("processing problem:", problem_name, "with crossover:", crossover_name, "and seed individual:", seed_type, "using algorithm:", algorithm_name)

                    pop_X = np.concatenate((seed_inds, random_populations[problem_name]), axis=0)
                    pop = t_sampling.do(problem, pop_size, seeds=pop_X)

                    algorithm = get_algorithm(
                        algorithm_name=algorithm_name,
                        n_obj=problem.n_obj,
                        pop_size=pop_size,
                        sampling=pop,crossover=crossovers[crossover_name],
                        mutation=T_Mutation(mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20), tracing_type=tracing_type, accumulate_mutations=True)
                        )

                    for i in range(31):#31 re-runs as usual
                        #set up callbacks:
                        additional_run_info = {
                            "run_number": i,
                            "crossover": crossover_name,
                            "problem_name": problem_name,
                            "algorithm_name": algorithm_name,
                            "seed_type": seed_type,
                            }
                        callbacks = [
                        # Counting_Impact_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="counting_impact_pop_do"+str(problem.n_obj)), #I need to save separately for each number of objectives
                        Counting_Impact_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=True, filename="counting_impact_opt_do"+str(problem.n_obj)),
                        Performance_Indicators_Callback(additional_run_info=additional_run_info, filename="performance_indicators_do"+str(problem.n_obj)),
                        Fitness_and_Ranks_Callback(additional_run_info=additional_run_info, n_obj=problem.n_obj, filename="fitness_and_ranks_do"+str(problem.n_obj)),
                        # Genome_Callback(additional_run_info=additional_run_info, n_var=problem.n_var, filename="genome_do"+str(problem.n_obj)), this data is unnessecary atm.
                        ]
                        callback = AccumulateCallbacks(collectors=callbacks)

                        #run the test
                        res = minimize(problem,
                                    algorithm,
                                    ('n_gen', n_gen),
                                    seed=i,#seed is the run number!
                                    verbose=False,
                                    callback=callback
                                    )
                        
                        #print output:
                        callback.finalize(output_folder)