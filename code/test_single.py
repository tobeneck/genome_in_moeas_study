import numpy as np
import pandas as pd

from os.path import exists

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


inds_optimal_and_worst = np.array([
    #the optimal individual:
    [0,0,0,0],
    #the (fitness wise) nadir point:
    [5,5,5,5],
])
# inds_f25 = np.array([#fitness 25: #this is essentially never chosen, as the worst fitness of the pop is already lower then 16
# [5,0,0,0],
# [np.sqrt(25/2),np.sqrt(25/2),0,0],
# [np.sqrt(25/3),np.sqrt(25/3),np.sqrt(25/3),0],
# [np.sqrt(25/4),np.sqrt(25/4),np.sqrt(25/4),np.sqrt(25/4)],
# ])
inds_f16 = np.array([#fitness 16:
    [4,0,0,0],
    [np.sqrt(16/2),np.sqrt(16/2),0,0],
    [np.sqrt(16/3),np.sqrt(16/3),np.sqrt(16/3),0],
    [np.sqrt(16/4),np.sqrt(16/4),np.sqrt(16/4),np.sqrt(16/4)],
])
inds_f9 = np.array([#fitness 9:
    [3,0,0,0],
    [np.sqrt(9/2),np.sqrt(9/2),0,0],
    [np.sqrt(9/3),np.sqrt(9/3),np.sqrt(9/3),0],
    [np.sqrt(9/4),np.sqrt(9/4),np.sqrt(9/4),np.sqrt(9/4)],
])
inds_f4 = np.array([#fitness 4:
    [2,0,0,0],
    [np.sqrt(2/2),np.sqrt(2/2),0,0],
    [np.sqrt(2/3),np.sqrt(2/3),np.sqrt(2/3),0],
    [np.sqrt(2/4),np.sqrt(2/4),np.sqrt(2/4),np.sqrt(2/4)],
])
inds_f1 = np.array([#fitness 1:
    [1,0,0,0],
    [np.sqrt(1/2),np.sqrt(1/2),0,0],
    [np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3),0],
    [np.sqrt(1/4),np.sqrt(1/4),np.sqrt(1/4),np.sqrt(1/4)],
])

print("dist f9, seed_type0", np.linalg.norm(inds_f9[0]-inds_optimal_and_worst[0]))
print("dist f9, seed_type1", np.linalg.norm(inds_f9[1]-inds_optimal_and_worst[0]))
print("dist f9, seed_type2", np.linalg.norm(inds_f9[2]-inds_optimal_and_worst[0]))
print("dist f9, seed_type3", np.linalg.norm(inds_f9[3]-inds_optimal_and_worst[0]))

output_filename = "test.csv"
#set the random seed!
np.random.seed(1)

#set up the algorithmic parameters:
n_gen=50
pop_size = 20
problem = Sphere(n_var = 4)
tracing_type = TracingTypes.TRACE_VECTOR
t_sampling = T_Sampling(sampling=FloatRandomSampling(), tracing_type=tracing_type)
t_crossover_combining = T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=tracing_type)
t_crossover_copying = T_Crossover(crossover=UniformCrossover(), tracing_type=tracing_type)
t_mutation = T_Mutation(mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20), tracing_type=tracing_type, accumulate_mutations=True)



problem_d4 = Sphere(n_var = 4)
random_inds_d4 = t_sampling.do(problem_d4, pop_size - 1).get("X") / 2 
problem_d8 = Sphere(n_var = 8)
random_inds_d8 = t_sampling.do(problem_d8, pop_size - 1).get("X") / 2.95
problem_d12 = Sphere(n_var = 12)
random_inds_d12 = t_sampling.do(problem_d12, pop_size - 1).get("X") / 3.3

print("d4 ", problem_d4.evaluate(random_inds_d4).max(), problem_d4.evaluate(random_inds_d4).mean(), problem_d4.evaluate(random_inds_d4).min())
print("d8 ",problem_d8.evaluate(random_inds_d8).max(),  problem_d8.evaluate(random_inds_d8).mean(), problem_d8.evaluate(random_inds_d8).min())
print("d12", problem_d12.evaluate(random_inds_d12).max(), problem_d12.evaluate(random_inds_d12).mean(), problem_d12.evaluate(random_inds_d12).min())

exit()

def get_callback(n_var, initial_popsize, tracing_type, additional_run_info=None):
    '''
    Sets up and returns the callback for the test.

    [0] = counting_impact
    [1] = so_fitness

    Parameters:
    -----------
    initial_popsize : int
        The size of the initial population.
    tracing_type : TraceType
        The tracing implementation used.
    additional_keys : dict
        A dictionary of additional keys used for logging the data.
    Returns:
    --------
    callback : tea_pymoo.callback.accumulated_callback
        The callback for the tests.
    '''
    callbacks = [
            Counting_Impact_Callback(additional_run_info = additional_run_info, initial_popsize = initial_popsize, tracing_type=tracing_type),
            Performance_Callback(additional_run_info=additional_run_info),
            Fitness_Callback(additional_run_info=additional_run_info),
            Genome_Callback(n_var = n_var, additional_run_info=additional_run_info)

        ]
    callback = AccumulateCallbacks(collectors=callbacks)
    return callback

def run_single_conf(seed, seed_fitness, seed_type, random_inds):
    '''
    generates the runs for one single seed.

    Parameters:
    -----------
    seed : np.array
        The seed for which to generate the test.
    seed_fitness : float
        The fitness of the seed.
    seed_type : int
        The type of the seed / the genome structure of the seed.
    random_inds : np.array
        The rest of the population for the test.
    '''

    print("processing seed with fitness", seed_fitness,"and type", seed_type)

    pop_X = np.concatenate(([seed], random_inds), axis=0)
    pop = t_sampling.do(problem, pop_size, seeds=pop_X)


    copy_algorithm = GA(#with copying crossover
        pop_size=pop_size,
        sampling=pop,
        crossover=t_crossover_copying,
        mutation=t_mutation,
        )
    combining_algorithm = GA(#with combining crossover
        pop_size=pop_size,
        sampling=pop,
        crossover=t_crossover_combining,
        mutation=t_mutation,
        )

    for i in range(31):

        #the copying crossover:
        additional_keys = {
            "run_number": i,
            "seed_fitness": seed_fitness,
            "seed_type": seed_type,
            "crossover_type": "copy"
            }
        callback = get_callback(problem.n_var, pop_size, tracing_type, additional_keys)
        res = minimize(problem,
                    copy_algorithm,
                    ('n_gen', n_gen),
                    seed=i,
                    verbose=False,
                    callback=callback)
        callback.finalize("test_out")
        #save_data_to_file(callback, seed_fitness, seed_type)

        #the combining crossover:
        additional_keys["crossover_type"] = "combine"
        callback = get_callback(problem.n_var, pop_size, tracing_type, additional_keys)
        res = minimize(problem,
                    combining_algorithm,
                    ('n_gen', n_gen),
                    seed=i,
                    verbose=False,
                    callback=callback)
        callback.finalize("test_out")
        #save_data_to_file(callback, seed_fitness, seed_type)
        

#actually run the tests:
random_inds = t_sampling.do(problem, pop_size - 1).get("X") / 2 #generate the other inds here to keep them equal. Fitness values between -2.5 to 2.5 to match the fitness of the seed individuals better!

for i in range(len(inds_f1)):
    run_single_conf(seed=inds_f1[i],seed_fitness=1,seed_type=i,random_inds=random_inds)
for i in range(len(inds_f4)):
    run_single_conf(seed=inds_f4[i],seed_fitness=4,seed_type=i,random_inds=random_inds)
for i in range(len(inds_f9)):
    run_single_conf(seed=inds_f9[i],seed_fitness=9,seed_type=i,random_inds=random_inds)
for i in range(len(inds_f16)):
    run_single_conf(seed=inds_f16[i],seed_fitness=16,seed_type=i,random_inds=random_inds)
# for i in range(len(inds_f25)):
#     run_single_conf(seed=inds_f25[i],seed_fitness=25,seed_type=i,random_inds=random_inds)

run_single_conf(seed=inds_optimal_and_worst[0],seed_fitness=0,seed_type=-1,random_inds=random_inds)
run_single_conf(seed=inds_optimal_and_worst[1],seed_fitness=100,seed_type=-1,random_inds=random_inds)





