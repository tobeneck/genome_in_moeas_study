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
pop_size = 20 #the overall popsize is 20, but we only need 19 individuals due to the seed! 

'''
NOTE: The same initial population is used for all problems, even though different problems with different upper and lower bounds are used.
'''
random_inds_d4 = (np.random.random((pop_size-1, 4)) - 0.5) * 10
random_inds_d8 = (np.random.random((pop_size-1, 8)) - 0.5) * 10
random_inds_d16 = (np.random.random((pop_size-1, 16)) - 0.5) * 10
random_inds_d32 = (np.random.random((pop_size-1, 32)) - 0.5) * 10

np.savetxt(out_path+"/random_inds_d4.csv", random_inds_d4, delimiter=",")
np.savetxt(out_path+"/random_inds_d8.csv", random_inds_d8, delimiter=",")
np.savetxt(out_path+"/random_inds_d16.csv", random_inds_d16, delimiter=",")
np.savetxt(out_path+"/random_inds_d32.csv", random_inds_d32, delimiter=",")

