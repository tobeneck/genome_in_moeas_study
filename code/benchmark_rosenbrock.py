'''
This file contains preliminary test data for the tests with the Rosenbrock function.
'''
import os

import pandas as pd
import numpy as np

from tea_pymoo.tracing.t_sampling import T_Sampling
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.tracing.t_mutation import T_Mutation
from tea_pymoo.tracing.tracing_types import TracingTypes
from tea_pymoo.callbacks.accumulated_callback import AccumulateCallbacks
from tea_pymoo.callbacks.soo.performance_callback import Performance_Callback
from tea_pymoo.callbacks.soo.fitness_callback import Fitness_Callback
from tea_pymoo.callbacks.general.counting_impact_callback import Counting_Impact_Callback

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover

from problems.rosenbrock import Rosenbrock

#check if the output path exists
out_path = "../data/rosenbrock_function_data"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#set the randon seed
np.random.seed(1)

#read the initial populations
in_path = "../data/initial_populations"
random_inds_d4 = np.loadtxt(in_path+"/random_inds_d4.csv", delimiter=",", dtype=float)
random_inds_d8 = np.loadtxt(in_path+"/random_inds_d8.csv", delimiter=",", dtype=float)
random_inds_d16 = np.loadtxt(in_path+"/random_inds_d16.csv", delimiter=",", dtype=float)
random_inds_d32 = np.loadtxt(in_path+"/random_inds_d32.csv", delimiter=",", dtype=float)


#read the seed individuals
seed_ind_d4_df = pd.read_csv("../data/seed_individuals/seed_individuals_Rosenbrock_4.csv")
seed_ind_d8_df = pd.read_csv("../data/seed_individuals/seed_individuals_Rosenbrock_8.csv")
seed_ind_d16_df = pd.read_csv("../data/seed_individuals/seed_individuals_Rosenbrock_16.csv")
seed_ind_d32_df = pd.read_csv("../data/seed_individuals/seed_individuals_Rosenbrock_32.csv")

#set up the 
dims = [4,8,16,32]
pop_size = 40
n_gen = 100
xu = 50
xl = -50
tracing_type = TracingTypes.TRACE_VECTOR

t_sampling = T_Sampling(FloatRandomSampling(), tracing_type=tracing_type)

crossovers = {
        "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=tracing_type),
        "SBX" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=tracing_type),
#        "LXg" : T_Crossover(crossover=LXg(), tracing_type=tracing_type),
#        "LXi" : T_Crossover(crossover=LXi(), tracing_type=tracing_type), #this should bias no optimal genetic material the most!
    }
random_populations = {
    4 : random_inds_d4,
    8 : random_inds_d8,
    16 : random_inds_d16,
    32 : random_inds_d32,
}
seed_ind_dfs = {
    4 : seed_ind_d4_df,
    8 : seed_ind_d8_df,
    16 : seed_ind_d16_df,
    32 : seed_ind_d32_df,
}
problems = {
    4 : Rosenbrock(n_var = 4, xu=xu, xl=xl),
    8 : Rosenbrock(n_var = 8, xu=xu, xl=xl),
    16 : Rosenbrock(n_var = 16, xu=xu, xl=xl),
    32 : Rosenbrock(n_var = 32, xu=xu, xl=xl),
}

#run the tests:
for dim in dims:#iterate over the dimensions
    problem = problems[dim]
    for crossover_name in crossovers:#iterate over the crossovers
        for index, seed_ind_data in seed_ind_dfs[dim].iterrows():#iterate over the seed individuals in this generation
            #extract the seed ind from the dataframe

            seed_individual = seed_ind_data[1:dim+1].to_numpy(dtype=float)#need to specify dtype, otherwise we will get an error

            #seed_fitness = get_fitness(problem, seed_individual) #TODO
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
                callback.finalize("../data/rosenbrock_function_data")
   