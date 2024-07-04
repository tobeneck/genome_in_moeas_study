'''
This file generates the initial populations used for the tests with the Sphere, Rastrigin, and Rosenbrock functions.
'''
import os
import numpy as np

from pymoo.operators.sampling.rnd import FloatRandomSampling

from test_setup import problems, pop_size


#check if the output path exists
out_path = "../data/initial_populations"
if not os.path.exists(out_path):
   os.makedirs(out_path)

#set the randon seed
np.random.seed(1)


'''
generate one initial population for each problem separately because of bounds.
'''
for problem_name in problems.keys():
   problem = problems[problem_name]
   sampling = FloatRandomSampling()

   n_obj = problem.n_obj
   if n_obj == 2:
      random_pop_size = pop_size - 3 #for the three seeds
   if n_obj == 3:
      random_pop_size = pop_size - 4 #for the four seeds<
   X = sampling._do(problem, random_pop_size)
   np.savetxt(out_path+"/random_pop_"+problem_name+"_ds"+str(problem.n_var)+"_do"+str(n_obj)+".csv", X, delimiter=",")
