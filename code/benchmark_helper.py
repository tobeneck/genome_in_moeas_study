from tea_pymoo.tracing.t_mutation import T_Mutation
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.callbacks.general.counting_impact_callback import Counting_Impact_Callback
from tea_pymoo.callbacks.soo.performance_callback import Performance_Callback
from tea_pymoo.callbacks.soo.fitness_callback import Fitness_Callback
from tea_pymoo.callbacks.accumulated_callback import AccumulateCallbacks
from tea_pymoo.tracing.tracing_types import TracingTypes
from tea_pymoo.tracing.t_sampling import T_Sampling

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover

import numpy as np

def run_tests(
        problems:dict,
        output_folder:str,
        seed_ind_dfs:dict,
        dims:np.array=[4,8,16,32],
        random_populations:dict = {
            4 : np.loadtxt("../data/initial_populations/random_inds_d4.csv", delimiter=",", dtype=float),
            8 : np.loadtxt("../data/initial_populations/random_inds_d8.csv", delimiter=",", dtype=float),
            16 : np.loadtxt("../data/initial_populations/random_inds_d16.csv", delimiter=",", dtype=float),
            32 : np.loadtxt("../data/initial_populations/random_inds_d32.csv", delimiter=",", dtype=float),
        },
        crossovers:dict= {
            "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR),
            "SBX" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=TracingTypes.TRACE_VECTOR),
        },
        n_gen:int=100,
        pop_size:int=40,
        tracing_type:TracingTypes=TracingTypes.TRACE_VECTOR,
        ):
    '''
    TODO!
    '''
    t_sampling = T_Sampling(FloatRandomSampling(), tracing_type=tracing_type)

    #run the tests:
    for dim in dims:#iterate over the dimensions
        problem = problems[dim]
        for crossover_name in crossovers:#iterate over the crossovers
            for index, seed_ind_data in seed_ind_dfs[dim].iterrows():#iterate over the seed individuals in this generation
                #extract the seed ind from the dataframe

                seed_individual = seed_ind_data[1:dim+1].to_numpy(dtype=float)#need to specify dtype, otherwise we will get an error

                percentage_of_optimal_genes = seed_ind_data["% optimal genes"]
                seed_quality = seed_ind_data["quality"]
                seed_fitness = seed_ind_data["fitness"]
                print("processing dim:", dim," seed quality:", seed_quality,"fitness", seed_fitness, " optimal genes:", percentage_of_optimal_genes," crossover:", crossover_name)

                pop_X = np.concatenate(([seed_individual], random_populations[dim]), axis=0)
                pop = t_sampling.do(problem, pop_size, seeds=pop_X)

                algorithm = GA(
                    pop_size=pop_size,
                    sampling=pop,
                    crossover=T_Crossover(crossover=crossovers[crossover_name], tracing_type=tracing_type),
                    mutation=T_Mutation(mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20), tracing_type=tracing_type, accumulate_mutations=True)
                    )

                for i in range(31):#31 re-runs as usual
                    #set up callbacks:
                    additional_run_info = {
                        "run_number": i,
                        "seed_fitness": seed_fitness,
                        "seed_quality": seed_quality,
                        "percentage_of_optimal_genes": percentage_of_optimal_genes,
                        "crossover": crossover_name,
                        "dim": dim,
                        "seed_closeness_search_space": seed_ind_data["close/far"],#TODO: rename
                        "seed_distance_search_space": seed_ind_data["euclidean distance"],#TODO: rename
                        }
                    callbacks = [
                    Counting_Impact_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="counting_impact_pop"),
                    Counting_Impact_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=True, filename="counting_impact_opt"),
                    Performance_Callback(additional_run_info=additional_run_info),
                    Fitness_Callback(additional_run_info=additional_run_info),
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