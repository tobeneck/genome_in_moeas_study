'''
This file generates the initial populations used for the tests with the Sphere, Rastrigin, and Rosenbrock functions.
'''
import os
import numpy as np

#check if the output path exists
out_path = "../data/initial_populations"
if not os.path.exists(out_path):
   os.makedirs(out_path)

#set the randon seed
np.random.seed(1)

#generate the initial populations
pop_size = 91
search_space_dims = [10]
objective_space_dims = [2, 3]

'''
NOTE: The same initial population is used for all problems
'''
for d_s in search_space_dims:
   for d_o in objective_space_dims:
      random_pop = (np.random.random((pop_size-d_o-1, d_s))) #range from 0.0 to (almoust) 1.0
      np.savetxt(out_path+"/random_pop_ds"+str(d_s)+"_do"+str(d_o)+".csv", random_pop, delimiter=",")