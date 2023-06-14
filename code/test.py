from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems.multi.zdt import ZDT1
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover

from tea_pymoo.tracing.t_sampling import TracingTypes, T_Sampling
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.tracing.t_mutation import T_Mutation

from problems.MACO import MultiAgentCoordinationProblem

problem = MultiAgentCoordinationProblem()

tracing_type = TracingTypes.TRACE_LIST
t_sampling = T_Sampling(sampling=FloatRandomSampling(), tracing_type=tracing_type)
t_crossover = T_Crossover(crossover=SimulatedBinaryCrossover(prob=0.9, eta=20), tracing_type=tracing_type)
t_mutation = T_Mutation(mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20), tracing_type=tracing_type, accumulate_mutations=True)


algorithm = NSGA2(
    pop_size=100,
    sampling=t_sampling,
    crossover=t_crossover,
    mutation=t_mutation
    )

res = minimize(problem,
               algorithm,
               ('n_gen', 3),
               seed=1,
               verbose=False)

print(res.F)

print(res.pop.get("T")[0][0].get(0).traceID)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()