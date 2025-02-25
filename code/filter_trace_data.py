import pandas as pd

d2_trace_df = pd.read_csv("../data/e_and_c_benchmark/counting_impact_inds_nds_do3.csv")
d2_trace_df = d2_trace_df[d2_trace_df.generation==100]

d2_trace_df.to_csv("../data/e_and_c_benchmark/counting_impact_inds_do3_gen100.csv", index=False)