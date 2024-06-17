#ideal and nadir points found:

import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

def get_2d_ref_point(df, H=91):
    '''
    Get the ideal and nadir points for a 2D problem. Dataframe needs to be filtered already.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the fitness values.
    H : int
        The pop size / number of reference directories used.
    '''
    ref_point = [df.f_1.max(), df.f_2.max()]
    ref_point =[value + 1/H for value in ref_point]
    return ref_point

def get_3d_ref_point(df, H=91):
    '''
    Get the ideal and nadir points for a 3D problem. Dataframe needs to be filtered already.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the fitness values.
    H : int
        The pop size / number of reference directories used.
    '''
    ref_point = [df.f_1.max(), df.f_2.max(), df.f_3.max()]
    ref_point =[value + 1/H for value in ref_point]
    return ref_point




#only compute HV for a few rows, otherwise it will take forever
first_gen = 1
final_gen = 100
gens = [
    first_gen,
    final_gen
]

### =============================================================================
### the 3D problems
### =============================================================================
performance_indicators_d3 =  pd.read_csv("../data/e_and_c_benchmark/performance_indicators_do3.csv")
performance_indicators_d3["hv_ref1"] = np.nan
performance_indicators_d3["hv_ref2"] = np.nan

fitness_d3 = pd.read_csv("../data/e_and_c_benchmark/fitness_and_ranks_do3.csv")
fitness_d3 = fitness_d3[fitness_d3.generation.isin(gens)]

#compute the ref points for each problem in a dict:
ref_points_first_gen = dict()
ref_points_final_gen = dict()
for problem_name in fitness_d3.problem_name.unique():
    problem_df = fitness_d3[fitness_d3.problem_name == problem_name]
    ref_points_first_gen[problem_name] = get_3d_ref_point(problem_df[problem_df.generation == first_gen])
    ref_points_final_gen[problem_name] = get_3d_ref_point(problem_df[problem_df.generation == final_gen])

#filter performance indicator df for specified generations:
filtered_performance_indicators_d3 = performance_indicators_d3[performance_indicators_d3.generation.isin(gens)]

for index, row in filtered_performance_indicators_d3.iterrows():
    current_fitness_df = fitness_d3[(fitness_d3.problem_name == row.problem_name) & (fitness_d3.seed_type == row.seed_type) & (fitness_d3.algorithm_name == row.algorithm_name) & (fitness_d3.run_number == row.run_number) & (fitness_d3.crossover == row.crossover) & (fitness_d3.generation == row.generation)]
    
    pop = current_fitness_df[["f_1", "f_2", "f_3"]].to_numpy()
    first_gen_hv = HV(ref_point=ref_points_first_gen[row.problem_name])
    performance_indicators_d3.at[index, "hv_ref1"] = first_gen_hv(pop)
    final_gen_hv = HV(ref_point=ref_points_final_gen[row.problem_name])
    performance_indicators_d3.at[index, "hv_ref2"] = first_gen_hv(pop)

    print("Computed row", index, "of", len(performance_indicators_d3), "d3", row.problem_name, row.seed_type, row.algorithm_name, row.run_number, row.crossover, row.generation, performance_indicators_d3.at[index, "hv_ref1"], performance_indicators_d3.at[index, "hv_ref1"])

performance_indicators_d3.to_csv("../data/e_and_c_benchmark/performance_indicators_do3_plus_HV.csv", index=False)

### =============================================================================
### the 2D problems
### =============================================================================
performance_indicators_d2 =  pd.read_csv("../data/e_and_c_benchmark/performance_indicators_do2.csv")
performance_indicators_d2["hv_ref1"] = np.nan
performance_indicators_d2["hv_ref2"] = np.nan

fitness_d2 = pd.read_csv("../data/e_and_c_benchmark/fitness_and_ranks_do2.csv")
fitness_d2 = fitness_d2[fitness_d2.generation.isin(gens)]

#compute the ref points for each problem in a dict:
ref_points_first_gen = dict()
ref_points_final_gen = dict()
for problem_name in fitness_d2.problem_name.unique():
    problem_df = fitness_d2[fitness_d2.problem_name == problem_name]
    ref_points_first_gen[problem_name] = get_2d_ref_point(problem_df[problem_df.generation == first_gen])
    ref_points_final_gen[problem_name] = get_2d_ref_point(problem_df[problem_df.generation == final_gen])

#filter performance indicator df for specified generations:
filtered_performance_indicators_d2 = performance_indicators_d2[performance_indicators_d2.generation.isin(gens)]

for index, row in filtered_performance_indicators_d2.iterrows():
    current_fitness_df = fitness_d2[(fitness_d2.problem_name == row.problem_name) & (fitness_d2.seed_type == row.seed_type) & (fitness_d2.algorithm_name == row.algorithm_name) & (fitness_d2.run_number == row.run_number) & (fitness_d2.crossover == row.crossover) & (fitness_d2.generation == row.generation)]
    
    pop = current_fitness_df[["f_1", "f_2"]].to_numpy()
    first_gen_hv = HV(ref_point=ref_points_first_gen[row.problem_name])
    performance_indicators_d2.at[index, "hv_ref1"] = first_gen_hv(pop)
    final_gen_hv = HV(ref_point=ref_points_final_gen[row.problem_name])
    performance_indicators_d2.at[index, "hv_ref2"] = first_gen_hv(pop)

    print("Computed row", index, "of", len(performance_indicators_d2), "d2", row.problem_name, row.seed_type, row.algorithm_name, row.run_number, row.crossover, row.generation, performance_indicators_d2.at[index, "hv_ref1"], performance_indicators_d2.at[index, "hv_ref1"])

performance_indicators_d2.to_csv("../data/e_and_c_benchmark/performance_indicators_do2_plus_HV.csv", index=False)