#this file contains the general definition of the problems and algorithms used throughout the experiments to avoid dupilication mistakes.
from pymoo_problems.moo.maco import MACO
from pymoo_problems.moo.uf import UF1, UF2, UF3, UF8, UF9, UF10
from pymoo_problems.moo.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pymoo_problems.moo.dtlz import DTLZ1, DTLZ2, DTLZ3


#generate the initial populations
pop_size = 91
n_var = 10

problems = {
    "MACO_b" : MACO(n_var = 10),#for MACO there are actually four pop sizes used in the paper, I chose 10.
    "MACO_p=-10" : MACO(n_var = 10, p=-10),
    "MACO_w=shallow" : MACO(n_var = 10, wtype="shallow"),
    "MACO_w=steep" : MACO(n_var = 10, wtype="steep"),
    "UF1" : UF1(n_var = 10),
    "UF2" : UF2(n_var = 10),
    "UF3" : UF3(n_var = 10),
    "UF8" : UF8(n_var = 10),
    "UF9" : UF9(n_var = 10),
    "UF10" : UF10(n_var = 10),
    "ZDT1" : ZDT1(n_var = 30),
    "ZDT2" : ZDT2(n_var = 30),
    "ZDT3" : ZDT3(n_var = 30),
    "ZDT4" : ZDT4(n_var = 10),
    "ZDT6" : ZDT6(n_var = 10),
    "DTLZ1" : DTLZ1(n_var = 7, n_obj = 3),
    "DTLZ2" : DTLZ2(n_var = 10, n_obj = 3),
    "DTLZ3" : DTLZ3(n_var = 10, n_obj = 3),
}