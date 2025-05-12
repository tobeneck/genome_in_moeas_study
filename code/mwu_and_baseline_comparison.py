import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu

def calc_baseline_comparison(performance_data_df) -> dict:
    baseline_comparison_data = {'problem_name': [],
                           'algorithm': [],
                           'generation': [],
                           'seed_type': [],
                           'average_igd+': [],
                            'average_gd': [],
                           'average_hv': [],
                           'average_igd+_diff': [],
                           'average_gd_diff':[],
                           'average_hv_diff': [],
                           'percentual_igd+_diff': [],
                            'percentual_gd_diff': [],
                            'percentual_hv_diff': [],
                            # 'roi_igd+': [],
                            # 'roi_gd': [],
                            # 'roi_hv': [],
                           'p_value_igd+': [],
                           'p_value_gd': [],
                           'p_value_hv': [],
                           'percentual_igd+_diff_text': [],
                            'percentual_gd_diff_text': [],
                            'percentual_hv_diff_text': [],
                            'average_igd+_text': [],
                            'average_gd_text': [],
                            'average_hv_text': [],
                          }
    
    problem_names = performance_data_df.problem_name.unique()
    seed_types = performance_data_df.seed_type.unique()
    algorithm_names = performance_data_df.algorithm_name.unique()
    generations = performance_data_df.generation.unique()
    #generations = [performance_data_df.generation.max()]

    for problem_name in problem_names:
        current_problem = performance_data_df.loc[performance_data_df["problem_name"] == problem_name]
        for algorithm_name in algorithm_names:
            current_algorithm = current_problem.loc[current_problem["algorithm_name"] == algorithm_name]
            for generation in generations:
                current_genedarion = current_algorithm.loc[current_algorithm["generation"] == generation]
                for seed_type in seed_types:
                    print(f"Calculating baseline comparison for {problem_name}, {algorithm_name}, {generation}, {seed_type}")
                    random_only = current_genedarion.loc[current_genedarion.seed_type == "r"]
                    seed_type_only = current_genedarion.loc[current_genedarion.seed_type == seed_type]

                    baseline_comparison_data["problem_name"].append(problem_name)
                    baseline_comparison_data["algorithm"].append(algorithm_name)
                    baseline_comparison_data["generation"].append(generation)
                    baseline_comparison_data["seed_type"].append(seed_type)
                    baseline_comparison_data["average_igd+"].append(seed_type_only["igd+"].mean())
                    baseline_comparison_data["average_gd"].append(seed_type_only["gd"].mean())
                    baseline_comparison_data["average_hv"].append(seed_type_only["hv_ref2"].mean())

                    baseline_comparison_data["average_igd+_diff"].append(random_only["igd+"].mean() - seed_type_only["igd+"].mean())
                    baseline_comparison_data["average_gd_diff"].append(random_only["gd"].mean() - seed_type_only["gd"].mean())
                    baseline_comparison_data["average_hv_diff"].append(seed_type_only["hv_ref2"].mean() - random_only["hv_ref2"].mean())

                    percentual_igd_plus_diff = (random_only["igd+"].mean() / seed_type_only["igd+"].mean()) - 1 #igd+ smaller values are better, so random / seed
                    if random_only["hv_ref2"].mean() == 0: #TODO: this is a wierd fix, I need to think aboud this...
                         percentual_hv_diff = np.nan
                    else:
                        percentual_hv_diff = (seed_type_only["hv_ref2"].mean() / random_only["hv_ref2"].mean())- 1 #hv larger values are better, so seed / random
                    
                    if seed_type_only["gd"].mean() == 0: #TODO: this is a wierd fix, I need to think aboud this...
                         percentual_gd_diff = np.inf #we found the optimum...
                    else:
                        percentual_gd_diff = (random_only["gd"].mean() / seed_type_only["gd"].mean()) - 1 #gd smaller values are better, so random / seed

                    baseline_comparison_data["percentual_igd+_diff"].append( percentual_igd_plus_diff ) 
                    baseline_comparison_data["percentual_gd_diff"].append( percentual_gd_diff ) 
                    baseline_comparison_data["percentual_hv_diff"].append( percentual_hv_diff ) 

                    p_value_igd_plus = mannwhitneyu(random_only["igd+"], seed_type_only["igd+"]).pvalue
                    p_value_gd = mannwhitneyu(random_only["gd"], seed_type_only["gd"]).pvalue
                    p_value_hv = mannwhitneyu(random_only["hv_ref2"], seed_type_only["hv_ref2"]).pvalue

                    baseline_comparison_data["p_value_igd+"].append(p_value_igd_plus)
                    baseline_comparison_data["p_value_gd"].append(p_value_gd)
                    baseline_comparison_data["p_value_hv"].append(p_value_hv)

                    igd_significant_text = "*" if p_value_igd_plus < 0.05 else ""
                    gd_significant_text = "*" if p_value_gd < 0.05 else ""
                    hv_significant_text = "*" if p_value_hv < 0.05 else ""

                    baseline_comparison_data["percentual_igd+_diff_text"].append( str(np.around(percentual_igd_plus_diff, 3)) + f" {igd_significant_text}" )
                    baseline_comparison_data["percentual_gd_diff_text"].append( str(np.around(percentual_gd_diff, 3)) + f" {gd_significant_text}" )
                    baseline_comparison_data["percentual_hv_diff_text"].append( str(np.around(percentual_hv_diff, 3)) + f" {hv_significant_text}" )

                    baseline_comparison_data["average_igd+_text"].append( str(np.around(seed_type_only["igd+"].mean(), 3)) + f"{igd_significant_text}" )
                    baseline_comparison_data["average_gd_text"].append( str(np.around(seed_type_only["gd"].mean(), 3)) + f"{gd_significant_text}" )
                    baseline_comparison_data["average_hv_text"].append( str(np.around(seed_type_only["hv_ref2"].mean(), 3)) + f"{hv_significant_text}" )
    return baseline_comparison_data


performance_indicators_d2 = pd.read_csv("../data/e_and_c_benchmark/performance_indicators_do2_plus_HV.csv")
df = pd.DataFrame(calc_baseline_comparison(performance_indicators_d2))
df.to_csv("../data/e_and_c_benchmark/baseline_comparison_data_d2.csv.tar.gz", index=False)

performance_indicators_d3 = pd.read_csv("../data/e_and_c_benchmark/performance_indicators_do3_plus_HV.csv")
df = pd.DataFrame(calc_baseline_comparison(performance_indicators_d3))
df.to_csv("../data/e_and_c_benchmark/baseline_comparison_data_d3.csv.tar.gz", index=False)