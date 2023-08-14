'''
This file only contains the first preliminary tests used to explore the data.
For the actual benchmarks used look into "benchmark_sphere.py".
'''

import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover

from tea_pymoo.tracing.tracing_types import TracingTypes
from tea_pymoo.tracing.t_sampling import T_Sampling
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.tracing.t_mutation import T_Mutation

from tea_pymoo.callbacks.general.counting_impact_callback import Counting_Impact_Callback
from tea_pymoo.callbacks.soo.performance_callback import Performance_Callback
from tea_pymoo.callbacks.soo.fitness_callback import Fitness_Callback
from tea_pymoo.callbacks.general.genome_callback import Genome_Callback
from tea_pymoo.callbacks.accumulated_callback import AccumulateCallbacks

from problems.Sphere import Sphere


#set the random seed!
np.random.seed(1)
    
def get_fitness(problem, ind):
    return problem.evaluate([ind])[0,0]
def get_percentage_optimal_genes(problem, ind):
    return np.sum(ind == problem._calc_pareto_set()) / problem.n_var


def generate_seed_individuals(n_var, initial_pop):
    '''
    Generates a numpy vector of seed individuals based on the number of 
    '''
    seed_individuals = np.zeros()
    return Null

#define the seed individuals:
inds_d4 = np.array([
    #the optimal individual:
    [0,0,0,0],
    #the (fitness wise) nadir point:
    [5,5,5,5],
    
    #fitness 16:
    [4,0,0,0],
    [np.sqrt(16/2),np.sqrt(16/2),0,0],
    [np.sqrt(16/3),np.sqrt(16/3),np.sqrt(16/3),0],
    [np.sqrt(16/4),np.sqrt(16/4),np.sqrt(16/4),np.sqrt(16/4)],

    #fitness 9:
    [3,0,0,0],
    [np.sqrt(9/2),np.sqrt(9/2),0,0],
    [np.sqrt(9/3),np.sqrt(9/3),np.sqrt(9/3),0],
    [np.sqrt(9/4),np.sqrt(9/4),np.sqrt(9/4),np.sqrt(9/4)],

    #fitness 4:
    [2,0,0,0],
    [np.sqrt(4/2),np.sqrt(4/2),0,0],
    [np.sqrt(4/3),np.sqrt(4/3),np.sqrt(4/3),0],
    [np.sqrt(4/4),np.sqrt(4/4),np.sqrt(4/4),np.sqrt(4/4)],

    #fitness 1:
    [1,0,0,0],
    [np.sqrt(1/2),np.sqrt(1/2),0,0],
    [np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3),0],
    [np.sqrt(1/4),np.sqrt(1/4),np.sqrt(1/4),np.sqrt(1/4)],
])

inds_d8 = np.array([
    #the optimal individual:
    [0,0,0,0,0,0,0,0],
    #the (fitness wise) nadir point:
    [5,5,5,5,5,5,5,5],
    
    #fitness 16:
    [np.sqrt(16/2),np.sqrt(16/2),0,0,0,0,0,0],
    [np.sqrt(16/4),np.sqrt(16/4),np.sqrt(16/4),np.sqrt(16/4),0,0,0,0],
    [np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),0,0],
    [np.sqrt(16/8),np.sqrt(16/8),np.sqrt(16/8),np.sqrt(16/8),np.sqrt(16/8),np.sqrt(16/8),np.sqrt(16/8),np.sqrt(16/8)],

    #fitness 9:
    [np.sqrt(9/2),np.sqrt(9/2),0,0,0,0,0,0],
    [np.sqrt(9/4),np.sqrt(9/4),np.sqrt(9/4),np.sqrt(9/4),0,0,0,0],
    [np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),0,0],
    [np.sqrt(9/8),np.sqrt(9/8),np.sqrt(9/8),np.sqrt(9/8),np.sqrt(9/8),np.sqrt(9/8),np.sqrt(9/8),np.sqrt(9/8)],

    #fitness 4:
    [np.sqrt(4/2),np.sqrt(4/2),0,0,0,0,0,0],
    [np.sqrt(4/4),np.sqrt(4/4),np.sqrt(4/4),np.sqrt(4/4),0,0,0,0],
    [np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),0,0],
    [np.sqrt(4/8),np.sqrt(4/8),np.sqrt(4/8),np.sqrt(4/8),np.sqrt(4/8),np.sqrt(4/8),np.sqrt(4/8),np.sqrt(4/8)],

    #fitness 1:
    [np.sqrt(1/2),np.sqrt(1/2),0,0,0,0,0,0],
    [np.sqrt(1/4),np.sqrt(1/4),np.sqrt(1/4),np.sqrt(1/4),0,0,0,0],
    [np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),0,0],
    [np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8)],
])

inds_d12 = np.array([
    #the optimal individual:
    [0,0,0,0,0,0,0,0,0,0,0,0],
    #the (fitness wise) nadir point:
    [5,5,5,5,5,5,5,5,5,5,5,5],
    
    #fitness 16:
    [np.sqrt(16/3),np.sqrt(16/3),np.sqrt(16/3),0,0,0,0,0,0,0,0,0],
    [np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),np.sqrt(16/6),0,0,0,0,0,0],
    [np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),np.sqrt(16/9),0,0,0],
    [np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12),np.sqrt(16/12)],

    #fitness 9:
    [np.sqrt(9/3),np.sqrt(9/3),np.sqrt(9/3),0,0,0,0,0,0,0,0,0],
    [np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),np.sqrt(9/6),0,0,0,0,0,0],
    [np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),np.sqrt(9/9),0,0,0],
    [np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12),np.sqrt(9/12)],

    #fitness 4:
    [np.sqrt(4/3),np.sqrt(4/3),np.sqrt(4/3),0,0,0,0,0,0,0,0,0],
    [np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),np.sqrt(4/6),0,0,0,0,0,0],
    [np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),np.sqrt(4/9),0,0,0],
    [np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12),np.sqrt(4/12)],

    #fitness 1:
    [np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3),0,0,0,0,0,0,0,0,0],
    [np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),np.sqrt(1/6),0,0,0,0,0,0],
    [np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),np.sqrt(1/9),0,0,0],
    [np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12),np.sqrt(1/12)],
])


n_gen=50
pop_size = 20

tracing_type = TracingTypes.TRACE_VECTOR
t_sampling = T_Sampling(sampling=FloatRandomSampling(), tracing_type=tracing_type)



problems = {
        4 : Sphere(n_var = 4),
        8 : Sphere(n_var = 8),
        12 : Sphere(n_var = 12)
    }

random_inds_d4 = t_sampling.do(problems[4], pop_size - 1).get("X") / 2 
random_inds_d8 = t_sampling.do(problems[8], pop_size - 1).get("X") / 2.95
random_inds_d12 = t_sampling.do(problems[12], pop_size - 1).get("X") / 3.3

crossovers = {
        "combining" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=tracing_type),
        "copying" : T_Crossover(crossover=UniformCrossover(), tracing_type=tracing_type)
    }
seed_individuals = {
    4 : inds_d4,
    8 : inds_d8,
    12 : inds_d12
}
random_populations = {
    4 : random_inds_d4,
    8 : random_inds_d8,
    12 : random_inds_d12
}

for dim in problems:
    problem=problems[dim]
    for crossover_name in crossovers.keys():
        for seed_individual in seed_individuals[dim]:
            seed_fitness = get_fitness(problem, seed_individual)
            percentage_of_optimal_genes = get_percentage_optimal_genes(problem, seed_individual)
            print("processing dim:", dim," fitness:", round(seed_fitness)," optimal genes:", percentage_of_optimal_genes," crossover:", crossover_name)

            pop_X = np.concatenate(([seed_individual], random_populations[dim]), axis=0)
            pop = t_sampling.do(problem, pop_size, seeds=pop_X)

            algorithm = GA(
                pop_size=pop_size,
                sampling=pop,
                crossover=crossovers[crossover_name],
                mutation=T_Mutation(mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20), tracing_type=tracing_type, accumulate_mutations=True)
                )

            for i in range(31):#31 re-runs as usual
                #set up callbacks:
                additional_run_info = {
                    "run_number": i,
                    "seed_fitness": round(seed_fitness),
                    "percentage_of_optimal_genes": percentage_of_optimal_genes,
                    "crossover_type": crossover_name,
                    "dim": dim
                    }
                callbacks = [
                Counting_Impact_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type),
                Performance_Callback(additional_run_info=additional_run_info),
                Fitness_Callback(additional_run_info=additional_run_info),
                #Genome_Callback(n_var = problem.n_var, additional_run_info=additional_run_info, filename="genome_"+str(dim)) #genome information would need to be saved separately for each dimension!
                ]
                callback = AccumulateCallbacks(collectors=callbacks)

                #run the test
                res = minimize(problem,
                            algorithm,
                            ('n_gen', n_gen),
                            seed=i,
                            verbose=False,
                            callback=callback)
                
                #print output:
                callback.finalize("../data/preliminary_test_data")