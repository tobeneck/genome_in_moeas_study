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
    #dtlz
    "dtlz1": DTLZ1(n_var=n_var, n_obj=3),
    "dtlz2": DTLZ2(n_var=n_var, n_obj=3),
    "dtlz3": DTLZ3(n_var=n_var, n_obj=3),

    #zdt
    "zdt1": ZDT1(n_var=n_var),
    "zdt2": ZDT2(n_var=n_var),
    "zdt3": ZDT3(n_var=n_var),

    #uf
    "uf1": UF1(n_var=n_var),
    "uf2": UF2(n_var=n_var),
    "uf3": UF3(n_var=n_var),

    #maco
    "maco_basic": MACO(n_var=n_var, p=-np.inf, weights=None, classes=None, ctype=None, wtype=None),
    "maco_p-norm": MACO(n_var=n_var, p=2, weights=None, classes=None, ctype=None, wtype=None),
    "maco_weights": MACO(n_var=n_var, p=-np.inf, weights=None, classes=None, ctype=None, wtype="steep"),
    "maco_classes": MACO(n_var=n_var, p=-np.inf, weights=None, classes=None, ctype="half", wtype=None),
}

'''
generate one initial population for each problem separately because of bounds.
'''
for problem_name in problems.keys():
   problem = problems[problem_name]
   sampling = FloatRandomSampling()
   X = sampling._do(problem, pop_size)
   print(X)
   np.savetxt(out_path+"/random_pop_"+problem_name+"_ds"+str(problem.n_var)+"_do"+str(problem.n_obj)+".csv", X, delimiter=",")
