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

def generate_sphere_individual(n_var: int, percentage_optimal_genes: float, target_fitness: float):
    number_optimal_genes = int(percentage_optimal_genes * n_var)
    number_not_optimal_genes = n_var - number_optimal_genes

    ind = np.zeros(n_var)
    ind[:number_not_optimal_genes] = np.sqrt(target_fitness/number_not_optimal_genes)

    return ind


def generate_seed_individuals(initial_pop: np.array, problem: Sphere, n_var: int):
    seed_inds = np.zeros((18, n_var))

    #set the utopia and dystopia points
    seed_inds[0] = np.zeros(n_var)
    seed_inds[1] = np.ones(n_var) * 5

    #calculate the target fitness of the quartiles
    initial_pop_fitness = problem.evaluate(initial_pop)
    best = np.min(initial_pop_fitness) / 2
    lower_quartile = np.quantile(initial_pop_fitness, 0.25)
    median = np.median(initial_pop_fitness)
    upper_quartile = np.quantile(initial_pop_fitness, 0.75)


    seed_inds[2] = generate_sphere_individual(n_var, 0.0, best)
    seed_inds[3] = generate_sphere_individual(n_var, 0.25, best)
    seed_inds[4] = generate_sphere_individual(n_var, 0.5, best)
    seed_inds[5] = generate_sphere_individual(n_var, 0.75, best)

    seed_inds[6] = generate_sphere_individual(n_var, 0.0, lower_quartile)
    seed_inds[7] = generate_sphere_individual(n_var, 0.25, lower_quartile)
    seed_inds[8] = generate_sphere_individual(n_var, 0.5, lower_quartile)
    seed_inds[9] = generate_sphere_individual(n_var, 0.75, lower_quartile)

    seed_inds[10] = generate_sphere_individual(n_var, 0.0, median)
    seed_inds[11] = generate_sphere_individual(n_var, 0.25, median)
    seed_inds[12] = generate_sphere_individual(n_var, 0.5, median)
    seed_inds[13] = generate_sphere_individual(n_var, 0.75, median)

    seed_inds[14] = generate_sphere_individual(n_var, 0.0, upper_quartile)
    seed_inds[15] = generate_sphere_individual(n_var, 0.25, upper_quartile)
    seed_inds[16] = generate_sphere_individual(n_var, 0.5, upper_quartile)
    seed_inds[17] = generate_sphere_individual(n_var, 0.75, upper_quartile)

    seed_qualitys = [
        "utopia",
        "dystopia",
        "best",
        "best",
        "best",
        "best",
        "lower quartile",
        "lower quartile",
        "lower quartile",
        "lower quartile",
        "median",
        "median",
        "median",
        "median",
        "upper quartile",
        "upper quartile",
        "upper quartile",
        "upper quartile",
    ]

    return seed_inds, seed_qualitys





n_gen=50
pop_size = 20

tracing_type = TracingTypes.TRACE_VECTOR
t_sampling = T_Sampling(sampling=FloatRandomSampling(), tracing_type=tracing_type)



problems = {
        4 : Sphere(n_var = 4),
        8 : Sphere(n_var = 8),
        16 : Sphere(n_var = 16),
        32 : Sphere(n_var = 32),
    }


'''
The generated initial population needs to be in tighter bounds as -5 to +5,
otherwise the seed individuals of a target fitness can not be generated in the specified bounds,
which leads to problems in the SBX and PM operators.
'''
random_inds_d4 = t_sampling.do(problems[4], pop_size - 1).get("X") * 0.34
random_inds_d8 = t_sampling.do(problems[8], pop_size - 1).get("X") * 0.34
random_inds_d16 = t_sampling.do(problems[16], pop_size - 1).get("X") * 0.34
random_inds_d32 = t_sampling.do(problems[32], pop_size - 1).get("X") * 0.34

crossovers = {
        "SBX" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=tracing_type),
        "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=tracing_type)
    }
random_populations = {
    4 : random_inds_d4,
    8 : random_inds_d8,
    16 : random_inds_d16,
    32 : random_inds_d32,
}

for dim in problems:
    problem=problems[dim]
    for crossover_name in crossovers.keys():
        seed_individuals, seed_qualitys = generate_seed_individuals(random_populations[dim], problem, dim)
        for seed_ind_index in range(len(seed_individuals)):
            seed_individual = seed_individuals[seed_ind_index]
            seed_quality = seed_qualitys[seed_ind_index]

            seed_fitness = get_fitness(problem, seed_individual)
            percentage_of_optimal_genes = get_percentage_optimal_genes(problem, seed_individual)
            print("processing dim:", dim," seed quality:", seed_quality," optimal genes:", percentage_of_optimal_genes," crossover:", crossover_name)

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
                    "seed_quality": seed_quality,
                    "percentage_of_optimal_genes": percentage_of_optimal_genes,
                    "crossover": crossover_name,
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
                            seed=i,#seed is the run number!
                            verbose=False,
                            callback=callback)
                
                #print output:
                callback.finalize("../data/sphere_function_data_out")