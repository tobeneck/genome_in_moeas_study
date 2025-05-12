import pandas as pd
from test_setup import max_gen


#trace data d2
print("processing counting_impact_inds_nds_do2")
d2_trace_df = pd.read_csv("../data/e_and_c_benchmark/counting_impact_inds_nds_do2.csv")
d2_trace_df = d2_trace_df[d2_trace_df.generation==max_gen]
d2_trace_df.to_csv("../data/e_and_c_benchmark/counting_impact_inds_do2_gen100.csv.tar.gz", index=False)

d2_trace_df = pd.read_csv("../data/e_and_c_benchmark/counting_impact_pop_nds_do2.csv")
d2_trace_df = d2_trace_df[d2_trace_df.generation==max_gen]
d2_trace_df.to_csv("../data/e_and_c_benchmark/counting_impact_pop_do2_gen100.csv.tar.gz", index=False)

#trace data d3
print("processing counting_impact_inds_nds_do3")
d3_trace_df = pd.read_csv("../data/e_and_c_benchmark/counting_impact_inds_nds_do3.csv")
d3_trace_df = d3_trace_df[d3_trace_df.generation==max_gen]
d3_trace_df.to_csv("../data/e_and_c_benchmark/counting_impact_inds_do3_gen100.csv.tar.gz", index=False)

d2_trace_df = pd.read_csv("../data/e_and_c_benchmark/counting_impact_pop_nds_do3.csv")
d2_trace_df = d2_trace_df[d2_trace_df.generation==max_gen]
d2_trace_df.to_csv("../data/e_and_c_benchmark/counting_impact_pop_do3_gen100.csv.tar.gz", index=False)

#data of inds data d2
print("processing counting_impact_pop_nds_do2")
d2_fitness_df = pd.read_csv("../data/e_and_c_benchmark/fitness_and_ranks_do2.csv")
d2_fitness_df = d2_fitness_df[d2_fitness_df.generation==max_gen]
d2_fitness_df.to_csv("../data/e_and_c_benchmark/fitness_and_ranks_do2_gen100.csv.tar.gz", index=False)

#data of inds data d3
print("processing counting_impact_pop_nds_do3")
d3_fitness_df = pd.read_csv("../data/e_and_c_benchmark/fitness_and_ranks_do3.csv")
d3_fitness_df = d3_fitness_df[d3_fitness_df.generation==max_gen]
d3_fitness_df.to_csv("../data/e_and_c_benchmark/fitness_and_ranks_do3_gen100.csv.tar.gz", index=False)

#performance data d2
print("processing performance_indicators_do2_plus_HV")
d2_performance_df = pd.read_csv("../data/e_and_c_benchmark/performance_indicators_do2_plus_HV.csv")
d2_fitness_df = d2_fitness_df[d2_fitness_df.generation==max_gen]
d2_fitness_df.to_csv("../data/e_and_c_benchmark/performance_indicators_do2_plus_HV_gen100.csv.tar.gz", index=False)

#performance data d3
print("processing performance_indicators_do3_plus_HV")
d3_performance_df = pd.read_csv("../data/e_and_c_benchmark/performance_indicators_do3_plus_HV.csv")
d3_performance_df = d3_performance_df[d3_performance_df.generation==max_gen]
d3_performance_df.to_csv("../data/e_and_c_benchmark/performance_indicators_do3_plus_HV_gen100.csv.tar.gz", index=False)

print("done!")