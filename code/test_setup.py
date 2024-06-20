#this file contains the general definition of the problems and algorithms used throughout the experiments to avoid dupilication mistakes.
from pymoo_problems.moo.maco import MACO
from pymoo_problems.moo.uf import UF1, UF2, UF3
from pymoo_problems.moo.zdt import ZDT1, ZDT2, ZDT3
from pymoo_problems.moo.dtlz import DTLZ1, DTLZ2, DTLZ3


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