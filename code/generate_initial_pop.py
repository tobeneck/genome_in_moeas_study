'''
This file generates the initial populations used for the tests with the Sphere, Rastrigin, and Rosenbrock functions.
'''
import os
import numpy as np

from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo_problems.moo.maco import MACO
from pymoo_problems.moo.uf import UF1, UF2, UF3
from pymoo_problems.moo.zdt import ZDT1, ZDT2, ZDT3
from pymoo_problems.moo.dtlz import DTLZ1, DTLZ2, DTLZ3

#check if the output path exists
out_path = "../data/initial_populations"
if not os.path.exists(out_path):
   os.makedirs(out_path)

#set the randon seed
np.random.seed(1)

#generate the initial populations
pop_size = 91
n_var = 10

problems = {
    "MACO_b" : MACO(n_var = n_var),
    "MACO_p=-10" : MACO(n_var = n_var, p=-10),
    "MACO_w=shallow" : MACO(n_var = n_var, wtype="shallow"),
    "MACO_w=steep" : MACO(n_var = n_var, wtype="steep"),
    "UF1" : UF1(n_var = n_var),
    "UF2" : UF2(n_var = n_var),
    "UF3" : UF3(n_var = n_var),
    "ZDT1" : ZDT1(n_var = n_var),
    "ZDT2" : ZDT2(n_var = n_var),
    "ZDT3" : ZDT3(n_var = n_var),
    "DTLZ1" : DTLZ1(n_var = n_var, n_obj = 3),
    "DTLZ2" : DTLZ2(n_var = n_var, n_obj = 3),
    "DTLZ3" : DTLZ3(n_var = n_var, n_obj = 3),
}

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
   print(X)
   np.savetxt(out_path+"/random_pop_"+problem_name+"_ds"+str(problem.n_var)+"_do"+str(n_obj)+".csv", X, delimiter=",")
